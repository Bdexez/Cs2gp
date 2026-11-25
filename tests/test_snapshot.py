import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from cs2gp.predict import Analyzer
from cs2gp.ratings import EloConfig
from cs2gp.pronostic import fit_decider_temperature
from cs2gp.sources import simulate_history


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = simulate_history(n_teams=8, years=.5, seed=11, start=dt.date(2023, 1, 1))
        cls.an = Analyzer(EloConfig(k_base=14), warmup_maps=4, momentum=.1).load_maps(cls.maps)
        cls.an.train(use_gbdt=False, l2=3, calibration_frac=.25, recency_halflife=400)
        cls.on = max(m.date for m in cls.maps) + dt.timedelta(days=1)

    def test_round_trip_preserves_both_models_and_prediction_without_fitting(self):
        a, b = self.an.known_teams()[:2]
        expected = self.an.predict(a, b, on=self.on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2gp"
            self.an.save(path)
            with patch.object(Analyzer, "train", side_effect=AssertionError("nouvel entrainement")), \
                    patch("cs2gp.model.MatchPredictor.fit", side_effect=AssertionError("refit")):
                restored = Analyzer.load(path)
                actual = restored.predict(a, b, on=self.on)
                self.assertEqual(fit_decider_temperature(restored), fit_decider_temperature(self.an))
            self.assertEqual(restored.maps, self.maps)
            self.assertEqual(actual, expected)
            self.assertEqual(restored.elo_config, self.an.fx.elo.cfg)
            for name, value in restored._training_options.items():
                self.assertEqual(value, getattr(self.an.match_model, name))
            for name in ("match", "map"):
                model = getattr(restored, f"{name}_model")
                original = getattr(self.an, f"{name}_model")
                X = getattr(self.an, f"{name}_ds").X
                np.testing.assert_array_equal(model.predict_proba(X), original.predict_proba(X))
            self.assertEqual(restored.resolve_team("MOUZ"), self.an.resolve_team("MOUZ"))

    def test_historical_prediction_retains_original_training_options(self):
        on = self.an.match_ds.dates[-60]
        a, b = self.an.known_teams()[:2]
        expected = self.an.predict(a, b, on=on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2gp"
            self.an.save(path)
            restored = Analyzer.load(path)
            self.assertEqual(restored.predict(a, b, on=on), expected)
            restored.save(path)
            self.assertEqual(Analyzer.load(path).predict(a, b, on=on), expected)

    def test_corrupt_history_is_rejected_before_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2gp"
            self.an.save(path)
            with ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            files["history.json"] += b" "
            with ZipFile(path, "w") as archive:
                for name, blob in files.items():
                    archive.writestr(name, blob)
            with patch("cs2gp.model.MatchPredictor.load", side_effect=AssertionError("trop tot")):
                with self.assertRaisesRegex(ValueError, "somme de controle.*history"):
                    Analyzer.load(path)

    def test_bundle_preserves_l2_grid_and_historical_retraining(self):
        an = Analyzer(warmup_maps=4, momentum=.1).load_maps(self.maps).train(
            use_gbdt=False, l2=17.0, l2_grid=(.25, 4.0), calibration_frac=.1, recency_halflife=120,
        )
        a, b = an.known_teams()[:2]
        on = an.match_ds.dates[-60]
        expected = an.predict(a, b, on=on)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.cs2gp"
            an.save(path)
            restored = Analyzer.load(path)
            self.assertEqual(restored._training_options["l2_grid"], (.25, 4.0))
            self.assertEqual(restored._training_options["l2"], 17.0)
            self.assertEqual(restored.match_model.l2, an.match_model.l2)
            self.assertEqual(restored.map_model.l2, an.map_model.l2)
            self.assertEqual(restored.predict(a, b, on=on), expected)
            restored.save(path)
            self.assertEqual(Analyzer.load(path).predict(a, b, on=on), expected)

    def test_legacy_model_json_uses_manifest_for_missing_training_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy-settings.cs2gp"
            self.an.save(path)
            with ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(files["manifest.json"])
            manifest["training_options"].pop("l2_grid")
            for name in ("match.json", "map.json"):
                payload = json.loads(files[name])
                payload["version"] = 1
                for option in ("use_gbdt", "calibration_frac", "recency_halflife", "l2_grid"):
                    payload.pop(option)
                payload["gbdt_weight"] = 0.0
                files[name] = json.dumps(payload).encode()
                manifest["sha256"][name] = hashlib.sha256(files[name]).hexdigest()
            files["manifest.json"] = json.dumps(manifest).encode()
            with ZipFile(path, "w") as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
            restored = Analyzer.load(path)
            for model in (restored.match_model, restored.map_model):
                self.assertEqual(model.calibration_frac, .25)
                self.assertEqual(model.recency_halflife, 400)
            a, b = self.an.known_teams()[:2]
            self.assertEqual(restored.predict(a, b, on=self.on), self.an.predict(a, b, on=self.on))

    def test_incompatible_schema_and_unknown_archive_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2gp"
            for mode in ("schema", "version", "path"):
                self.an.save(path)
                with ZipFile(path) as archive:
                    files = {name: archive.read(name) for name in archive.namelist()}
                manifest = json.loads(files["manifest.json"])
                if mode == "schema":
                    manifest["schemas"]["match"].reverse()
                elif mode == "version":
                    manifest["version"] = 99
                else:
                    files["../unexpected"] = b"unexpected"
                files["manifest.json"] = json.dumps(manifest).encode()
                with ZipFile(path, "w") as archive:
                    for name, blob in files.items():
                        archive.writestr(name, blob)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    Analyzer.load(path)
            self.assertFalse((path.parent.parent / "unexpected").exists())

    def test_failed_save_preserves_previous_artifact_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analyzer.cs2gp"
            self.an.save(path)
            previous = path.read_bytes()
            with patch("cs2gp.snapshot.os.replace", side_effect=OSError("disque indisponible")):
                with self.assertRaises(OSError):
                    self.an.save(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_legacy_bundle_still_loads_and_calibration_can_be_built_on_demand(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.cs2gp"
            self.an.save(path)
            with ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(files["manifest.json"])
            manifest["version"] = 1
            del manifest["sha256"]["calibration.json"]
            del files["calibration.json"]
            files["manifest.json"] = json.dumps(manifest).encode()
            with ZipFile(path, "w") as archive:
                for name, blob in files.items():
                    archive.writestr(name, blob)
            with patch("cs2gp.model.MatchPredictor.fit", side_effect=AssertionError("refit")):
                restored = Analyzer.load(path)
                self.assertIsNone(restored._calibration_history)
                a, b = self.an.known_teams()[:2]
                self.assertEqual(restored.predict(a, b, on=self.on), self.an.predict(a, b, on=self.on))
            self.assertEqual(fit_decider_temperature(restored), fit_decider_temperature(self.an))


if __name__ == "__main__":
    unittest.main()
