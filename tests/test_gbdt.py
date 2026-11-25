import datetime as dt
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from cs2gp.model import MatchPredictor
from cs2gp.csvio import atomic_write_bytes
from cs2gp.predict import Analyzer
from cs2gp.sources import simulate_history


class GBDTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("sklearn") is None:
            if os.environ.get("CS2GP_REQUIRE_GBDT") == "1":
                raise RuntimeError("scikit-learn est requis pour cette verification")
            raise unittest.SkipTest("option GBDT non installee")
        rng = np.random.default_rng(91)
        cls.X = rng.normal(size=(600, 4))
        y = (cls.X[:, 0] + rng.normal(size=600)*.7 > 0).astype(float)
        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(600)]
        cls.model = MatchPredictor(use_gbdt=True).fit(cls.X, y, dates, ("a", "b", "c", "d"))
        if cls.model.gbdt is None:
            raise AssertionError("le GBDT doit reellement etre entraine")

    def test_actual_ensemble_is_symmetric(self):
        np.testing.assert_allclose(self.model.predict_proba(self.X) +
                                   self.model.predict_proba(-self.X), 1, atol=1e-12)

    def test_ensemble_save_load_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            loaded = MatchPredictor.load(path)
            self.assertIsNotNone(loaded.gbdt)
            np.testing.assert_allclose(loaded.predict_proba(self.X),
                                       self.model.predict_proba(self.X), atol=1e-12)

    def test_missing_ensemble_component_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            (path.parent / json.loads(path.read_text())["gbdt_file"]).unlink()
            with self.assertRaisesRegex(ValueError, "GBDT.*manquant"):
                MatchPredictor.load(path)

    def test_mismatched_ensemble_component_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            blob = path.parent / json.loads(path.read_text())["gbdt_file"]
            blob.write_bytes(blob.read_bytes() + b"different artifact")
            with self.assertRaisesRegex(ValueError, "GBDT incoherent"):
                MatchPredictor.load(path)

    def new_generation(self):
        model = copy.deepcopy(self.model)
        model.logistic.weights *= .8
        model.gbdt.set_params(random_state=17)
        return model

    def test_failed_json_publication_preserves_previous_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            previous = path.read_bytes()
            component = path.parent / json.loads(previous)["gbdt_file"]
            previous_component = component.read_bytes()

            def interrupted(target, data):
                if Path(target) == path:
                    raise OSError("interruption avant publication")
                atomic_write_bytes(target, data)

            with patch("cs2gp.model.atomic_write_bytes", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "interruption"):
                    self.new_generation().save(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(component.read_bytes(), previous_component)
            np.testing.assert_array_equal(MatchPredictor.load(path).predict_proba(self.X),
                                          self.model.predict_proba(self.X))

    def test_reader_observes_complete_old_generation_until_json_is_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            expected = self.model.predict_proba(self.X)
            observed = []

            def publish(target, data):
                atomic_write_bytes(target, data)
                if Path(target).suffix == ".pkl":
                    observed.append(MatchPredictor.load(path).predict_proba(self.X))

            new = self.new_generation()
            with patch("cs2gp.model.atomic_write_bytes", side_effect=publish):
                new.save(path)
            self.assertEqual(len(observed), 1)
            np.testing.assert_array_equal(observed[0], expected)
            np.testing.assert_array_equal(MatchPredictor.load(path).predict_proba(self.X),
                                          new.predict_proba(self.X))
            self.assertFalse(np.array_equal(new.predict_proba(self.X), expected))

    def test_same_content_reuses_one_component_and_json_can_be_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            name = json.loads(path.read_text())["gbdt_file"]
            self.model.save(path)
            self.assertEqual(json.loads(path.read_text())["gbdt_file"], name)
            self.assertEqual(len(list(path.parent.glob("*.pkl"))), 1)
            renamed = path.with_name("renamed.json")
            path.rename(renamed)
            np.testing.assert_array_equal(MatchPredictor.load(renamed).predict_proba(self.X),
                                          self.model.predict_proba(self.X))

    def test_legacy_fixed_component_name_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            self.model.save(path)
            payload = json.loads(path.read_text())
            component = path.parent / payload.pop("gbdt_file")
            path.with_suffix(".gbdt.pkl").write_bytes(component.read_bytes())
            payload["version"] = 1
            for with_digest in (True, False):
                if not with_digest:
                    payload.pop("gbdt_sha256")
                path.write_text(json.dumps(payload))
                np.testing.assert_array_equal(MatchPredictor.load(path).predict_proba(self.X),
                                              self.model.predict_proba(self.X))

    def test_invalid_component_reference_is_rejected_before_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "match.json"
            self.model.save(path)
            original = json.loads(path.read_text())
            names = (None, "../other.pkl", "/tmp/other.pkl", "..\\other.pkl",
                     "gbdt-" + "0" * 64 + ".pkl")
            for name in names:
                path.write_text(json.dumps({**original, "gbdt_file": name}))
                with self.subTest(name=name), \
                        patch("cs2gp.model.pickle.loads", side_effect=AssertionError("trop tot")), \
                        self.assertRaisesRegex(ValueError, "reference.*GBDT"):
                    MatchPredictor.load(path)

    def test_complete_analyzer_snapshot_keeps_the_actual_ensemble(self):
        maps = simulate_history(n_teams=8, years=1, seed=11, start=dt.date(2023, 1, 1))
        an = Analyzer().load_maps(maps).train(use_gbdt=True)
        self.assertIsNotNone(an.match_model.gbdt)
        on = max(m.date for m in maps) + dt.timedelta(days=1)
        a, b = an.known_teams()[:2]
        expected = an.predict(a, b, on=on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2gp"
            an.save(path)
            with patch.object(MatchPredictor, "fit", side_effect=AssertionError("refit")):
                restored = Analyzer.load(path)
                self.assertIsNotNone(restored.match_model.gbdt)
                self.assertIsNone(restored.map_model.gbdt)
                self.assertEqual(restored.predict(a, b, on=on), expected)
            for version in (None, "incompatible"):
                with patch("cs2gp.snapshot._sklearn_version", return_value=version):
                    with self.assertRaisesRegex(ValueError, "meme version de scikit-learn"):
                        Analyzer.load(path)

    def test_legacy_bundle_with_fixed_component_name_keeps_actual_ensemble(self):
        maps = simulate_history(n_teams=8, years=1, seed=11, start=dt.date(2023, 1, 1))
        an = Analyzer().load_maps(maps).train(use_gbdt=True)
        self.assertIsNotNone(an.match_model.gbdt)
        a, b = an.known_teams()[:2]
        on = max(m.date for m in maps) + dt.timedelta(days=1)
        expected = an.predict(a, b, on=on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.cs2gp"
            an.save(path)
            with ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(files.pop("manifest.json"))
            manifest["training_options"].pop("l2_grid")
            for name in ("match", "map"):
                payload = json.loads(files[f"{name}.json"])
                payload["version"] = 1
                if payload["has_gbdt"]:
                    files[f"{name}.gbdt.pkl"] = files.pop(payload.pop("gbdt_file"))
                for option in ("use_gbdt", "calibration_frac", "recency_halflife", "l2_grid"):
                    payload.pop(option)
                files[f"{name}.json"] = json.dumps(payload).encode()
            manifest["sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
            with ZipFile(path, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                for name, data in files.items():
                    archive.writestr(name, data)
            with patch.object(MatchPredictor, "fit", side_effect=AssertionError("refit")):
                restored = Analyzer.load(path)
                self.assertIsNotNone(restored.match_model.gbdt)
                self.assertEqual(restored.predict(a, b, on=on), expected)


if __name__ == "__main__":
    unittest.main()
