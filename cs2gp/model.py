"""Modele de prediction.

Coeur : regression logistique sans constante, entrainee par IRLS avec ridge,
sur des features antisymetriques. L'effet de l'augmentation miroir
(X, y) -> (-X, 1-y) est calcule sans copier les donnees. Sans constante,
le modele verifie exactement p(A|B) = 1 - p(B|A) : aucun
biais d'ordre des equipes, ce qui est la premiere source d'erreur silencieuse
dans ce genre de systeme.

Option : un GBDT (scikit-learn) capte les non-linearites ; on melange les deux
en espace logit, puis on recalibre par temperature sur un holdout temporel.
Une probabilite mal calibree est pire qu'inutile des qu'on la compare a une cote.
"""

from __future__ import annotations

import datetime as dt
import json
import hashlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .csvio import atomic_write_bytes
from .metrics import log_loss
from .temporal import calibration_boundary, validate_dates
from .validation import (binary_labels, feature_matrix, finite_number,
                         observation_weights, positive_integer)

RECENCY_HALFLIFE_DAYS = 540.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def recency_weights(dates: list[dt.date], halflife: float = RECENCY_HALFLIFE_DAYS) -> np.ndarray:
    """Le meta CS2 bouge : un match d'il y a deux ans en dit moins que celui d'hier."""
    finite_number("halflife", halflife, strict=True)
    if not dates:
        return np.ones(0)
    last = max(dates)
    ages = np.array([(last - d).days for d in dates], dtype=float)
    return 0.5 ** (ages / halflife)


@dataclass
class LogisticModel:
    """Regression logistique ridge, sans constante, resolue par IRLS."""

    l2: float = 2.0
    max_iter: int = 60
    tol: float = 1e-8
    residualize: bool = True
    weights: np.ndarray | None = None
    scales: np.ndarray | None = None
    resid: np.ndarray | None = None
    cov: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()

    def _fit_residualizer(self, X: np.ndarray, w: np.ndarray) -> None:
        """Projette chaque colonne hors de la colonne 0 (l'ecart de niveau).

        Toutes les features sont fortement correlees a `elo_diff` : forme, pool,
        calendrier disent en grande partie la meme chose. Sans decorrelation, le
        ridge repartit ce signal commun au hasard entre les colonnes et les
        coefficients deviennent ininterpretables (`elo_diff` peut meme ressortir
        negatif). La transformation est lineaire et inversible : elle ne change
        pas ce que le modele peut representer, seulement la facon dont la
        penalite repartit le credit. Ce qui reste dans chaque colonne est sa
        contribution *nette* de l'ecart de niveau.
        """
        if not self.residualize or X.shape[1] < 2:
            self.resid = np.zeros(max(0, X.shape[1] - 1))
            return
        base = X[:, 0]
        w_base = w * base
        denom = float(base @ w_base)
        if denom < 1e-12:
            self.resid = np.zeros(X.shape[1] - 1)
            return
        self.resid = np.array([float(w_base @ X[:, j]) / denom for j in range(1, X.shape[1])])

    def _apply_residual(self, X: np.ndarray) -> np.ndarray:
        if self.resid is None or len(self.resid) == 0:
            return X
        out = np.array(X, dtype=float, copy=True)
        out[:, 1:] -= np.outer(out[:, 0], self.resid)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticModel":
        y = binary_labels(y)
        X = feature_matrix(X, rows=len(y))
        w = observation_weights(sample_weight, len(y))
        finite_number("l2", self.l2)
        finite_number("tol", self.tol, strict=True)
        positive_integer("max_iter", self.max_iter)
        if self.feature_names and len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names doit nommer chaque colonne de X")
        self._fit_residualizer(X, w)
        X = self._apply_residual(X)
        # Mise a l'echelle sans centrage : centrer casserait l'antisymetrie.
        self.scales = np.maximum(np.sqrt(np.average(X**2, axis=0, weights=w)), 1e-6)
        Xs = X / self.scales

        # sigmoid(-z) = 1 - sigmoid(z) : chaque ligne miroir apporte les
        # memes gradient et Hessienne. Le facteur 2 preserve la penalite L2
        # historique sans materialiser un second exemplaire du dataset.
        beta = np.zeros(Xs.shape[1])
        ridge = self.l2 * np.eye(Xs.shape[1])
        for _ in range(self.max_iter):
            p = sigmoid(Xs @ beta)
            s = w * np.clip(p * (1 - p), 1e-9, None)
            grad = 2.0 * (Xs.T @ (w * (y - p))) - self.l2 * beta
            hess = 2.0 * ((Xs.T * s) @ Xs) + ridge
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            beta += step
            if np.max(np.abs(step)) < self.tol:
                break
        self.weights = beta
        # Matrice d'information de Fisher sur l'echantillon non dedouble -> covariance asymptotique.
        p_orig = sigmoid(Xs @ beta)
        s_orig = w * np.clip(p_orig * (1.0 - p_orig), 1e-9, None)
        fisher = (Xs.T * s_orig) @ Xs + ridge
        try:
            self.cov = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            self.cov = np.linalg.pinv(fisher)
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.scales is None:
            raise RuntimeError("modele non entraine")
        X = feature_matrix(np.atleast_2d(X), columns=len(self.weights))
        Xr = self._apply_residual(X)
        return (Xr / self.scales) @ self.weights

    def decision_se(self, X: np.ndarray) -> np.ndarray:
        """Erreur type analytique du logit (approximation de Laplace / Fisher)."""
        if self.weights is None or self.scales is None or self.cov is None:
            raise RuntimeError("covariance indisponible (modele non entraine ou ancien format)")
        X = feature_matrix(np.atleast_2d(X), columns=len(self.weights))
        Xs = self._apply_residual(X) / self.scales
        var = np.einsum("ij,jk,ik->i", Xs, self.cov, Xs)
        return np.sqrt(np.maximum(var, 0.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision(X))

    def coefficients(self) -> list[tuple[str, float]]:
        """Poids ramenes a l'echelle des features standardisees (comparables entre eux)."""
        if self.weights is None:
            return []
        names = self.feature_names or tuple(f"f{i}" for i in range(len(self.weights)))
        return sorted(zip(names, self.weights.tolist()), key=lambda kv: -abs(kv[1]))

    def feature_diagnostics(self) -> list[dict]:
        """Poids standardises, erreurs types de Fisher et z-scores de Wald par feature."""
        if self.weights is None or self.scales is None:
            raise RuntimeError("modele non entraine")
        names = self.feature_names or tuple(f"f{i}" for i in range(len(self.weights)))
        ses = (
            np.sqrt(np.maximum(np.diag(self.cov), 0.0)).tolist()
            if self.cov is not None else [None] * len(self.weights)
        )
        rows = []
        for idx, (name, weight, scale, se) in enumerate(zip(names, self.weights.tolist(), self.scales.tolist(), ses)):
            z = (weight / se) if (se is not None and se > 1e-12) else None
            resid_proj = 0.0 if (idx == 0 or self.resid is None or idx - 1 >= len(self.resid)) else float(self.resid[idx - 1])
            rows.append({
                "feature": name,
                "weight": float(weight),
                "se": None if se is None else float(se),
                "z_score": None if z is None else float(z),
                "significant": bool(z is not None and abs(z) > 1.96),
                "scale": float(scale),
                "resid_proj": resid_proj,
            })
        rows.sort(key=lambda r: -(abs(r["z_score"]) if r["z_score"] is not None else abs(r["weight"])))
        return rows

    def condition_number(self) -> float | None:
        """Conditionnement numerique de la matrice de covariance de Fisher."""
        if self.cov is None:
            return None
        val = float(np.linalg.cond(self.cov))
        return val if math.isfinite(val) else None


def _fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Calibration par temperature : preserve l'ordre et la symetrie, 1 parametre."""
    if len(y) < 30 or len(np.unique(y)) < 2:
        return 1.0
    lo, hi = 0.3, 4.0
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if log_loss(y, sigmoid(z / m1)) < log_loss(y, sigmoid(z / m2)):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def select_l2(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[dt.date] | None = None,
    grid: tuple[float, ...] = (0.5, 2.0, 8.0, 25.0, 80.0),
    val_frac: float = 0.2,
    recency_halflife: float = RECENCY_HALFLIFE_DAYS,
    feature_names: tuple[str, ...] = (),
) -> dict:
    """Selection chronologique de la penalite ridge L2 sur la fin du jeu d'entrainement.

    Ne touche jamais a la periode d'evaluation ni au holdout de calibration.
    """
    y = binary_labels(y)
    X = feature_matrix(X, rows=len(y))
    finite_number("val_frac", val_frac, strict=True)
    if val_frac >= 1.0:
        raise ValueError("val_frac doit etre dans ]0, 1[")
    finite_number("recency_halflife", recency_halflife, strict=True)
    if not grid:
        raise ValueError("grid doit contenir au moins une valeur de l2")
    for v in grid:
        finite_number("l2", v)
    if dates is not None:
        if len(dates) != len(y):
            raise ValueError("une date est requise pour chaque resultat")
        validate_dates(dates)

    n_val = int(len(y) * val_frac)
    n_fit = len(y) - n_val
    if dates and n_val:
        n_fit = calibration_boundary(dates, n_fit, minimum=30)
        n_val = len(y) - n_fit
    if n_fit < 30 or n_val < 20 or len(np.unique(y[:n_fit])) < 2:
        return {"best_l2": float(grid[0]), "scores": [], "n_train": n_fit, "n_val": n_val}

    Xf, yf = X[:n_fit], y[:n_fit]
    Xv, yv = X[n_fit:], y[n_fit:]
    w = recency_weights(dates[:n_fit], recency_halflife) if dates else None
    scores: list[tuple[float, float]] = []
    for v in grid:
        mdl = LogisticModel(l2=float(v), feature_names=feature_names).fit(Xf, yf, sample_weight=w)
        ll = log_loss(yv, mdl.predict_proba(Xv))
        scores.append((float(v), float(ll)))
    best_l2 = min(scores, key=lambda kv: kv[1])[0]
    return {"best_l2": best_l2, "scores": scores, "n_train": n_fit, "n_val": n_val}


@dataclass
class MatchPredictor:
    """Ensemble logistique (+ GBDT optionnel) avec calibration temperature."""

    l2: float = 2.0
    use_gbdt: bool = True
    gbdt_weight: float = 0.35
    calibration_frac: float = 0.2
    recency_halflife: float = RECENCY_HALFLIFE_DAYS
    l2_grid: tuple[float, ...] | None = None
    logistic: LogisticModel = field(default_factory=LogisticModel)
    gbdt: object | None = None
    temperature: float = 1.0
    feature_names: tuple[str, ...] = ()
    trained_on: int = 0
    last_date: str = ""

    def __post_init__(self) -> None:
        self.validate_config()

    def validate_config(self) -> None:
        finite_number("l2", self.l2)
        finite_number("gbdt_weight", self.gbdt_weight, maximum=1.0)
        finite_number("recency_halflife", self.recency_halflife, strict=True)
        finite_number("calibration_frac", self.calibration_frac)
        if self.calibration_frac >= 1:
            raise ValueError("calibration_frac doit etre dans [0, 1[")
        if not isinstance(self.use_gbdt, bool):
            raise ValueError("use_gbdt doit etre un booleen")
        if self.l2_grid is not None:
            if not self.l2_grid:
                raise ValueError("l2_grid doit contenir au moins une valeur")
            for v in self.l2_grid:
                finite_number("l2_grid", v)

    # ------------------------------------------------------------------- train

    def fit(self, X: np.ndarray, y: np.ndarray, dates: list[dt.date] | None = None,
            feature_names: tuple[str, ...] = ()) -> "MatchPredictor":
        y = binary_labels(y)
        X = feature_matrix(X, rows=len(y))
        self.validate_config()
        if len(y) < 50:
            raise ValueError(f"trop peu de matchs pour entrainer ({len(y)}), il en faut ~50+")
        if dates is not None:
            if len(dates) != len(y):
                raise ValueError("une date est requise pour chaque resultat")
            validate_dates(dates)
        self.feature_names = feature_names or self.feature_names
        if self.feature_names and len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names doit nommer chaque colonne de X")
        self.trained_on = len(y)
        self.last_date = max(dates).isoformat() if dates else ""

        # Holdout temporel (les derniers matchs) pour la calibration.
        n_cal = int(len(y) * self.calibration_frac)
        n_cal = 0 if n_cal < 40 else n_cal
        n_fit = len(y) - n_cal
        if dates and n_cal:
            n_fit = calibration_boundary(dates, n_fit)
            n_cal = len(y) - n_fit
        Xf, yf = X[:n_fit], y[:n_fit]
        dates_fit = dates[:n_fit] if dates else None
        if self.l2_grid is not None:
            chosen = select_l2(
                Xf, yf, dates_fit, grid=tuple(self.l2_grid),
                recency_halflife=self.recency_halflife, feature_names=self.feature_names,
            )
            self.l2 = chosen["best_l2"]
        w = recency_weights(dates_fit, self.recency_halflife) if dates_fit else None

        self.logistic = LogisticModel(l2=self.l2, feature_names=self.feature_names)
        self.logistic.fit(Xf, yf, sample_weight=w)

        self.gbdt = self._fit_gbdt(Xf, yf, w) if self.use_gbdt else None

        if n_cal:
            z = self._raw_decision(X[n_fit:])
            self.temperature = _fit_temperature(z, y[n_fit:])
        else:
            self.temperature = 1.0
        return self

    def _fit_gbdt(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None):
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError:
            return None
        if len(y) < 400:  # sous ce volume le GBDT sur-apprend plus qu'il n'apporte
            return None
        Xa = np.vstack([X, -X])
        ya = np.concatenate([y, 1.0 - y])
        wa = np.concatenate([w, w]) if w is not None else None
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=250, learning_rate=0.05,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
        clf.fit(Xa, ya, sample_weight=wa)
        return clf

    # ----------------------------------------------------------------- predict

    def _gbdt_proba(self, X: np.ndarray) -> np.ndarray:
        # Symetrisation explicite : le GBDT n'est pas antisymetrique par nature.
        p_fwd = self.gbdt.predict_proba(X)[:, 1]
        p_rev = self.gbdt.predict_proba(-X)[:, 1]
        return 0.5 * (p_fwd + (1.0 - p_rev))

    def _raw_decision(self, X: np.ndarray) -> np.ndarray:
        z = self.logistic.decision(X)
        if self.gbdt is not None:
            z = (1 - self.gbdt_weight) * z + self.gbdt_weight * logit(self._gbdt_proba(X))
        return z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return sigmoid(self._raw_decision(X) / max(self.temperature, 1e-6))

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def predict_interval(self, x: np.ndarray, z_score: float = 1.959963984540054) -> tuple[float, float]:
        """Intervalle de confiance analytique sur la probabilite predite (defaut 95%)."""
        finite_number("z_score", z_score, strict=True)
        row = np.asarray(x, dtype=float).reshape(1, -1)
        z = float(self._raw_decision(row)[0])
        weight_log = (1.0 - self.gbdt_weight) if self.gbdt is not None else 1.0
        se = float(self.logistic.decision_se(row)[0]) * weight_log
        temp = max(self.temperature, 1e-6)
        lo = float(sigmoid((z - z_score * se) / temp))
        hi = float(sigmoid((z + z_score * se) / temp))
        return lo, hi

    def explain(self, x: np.ndarray, top: int = 6) -> list[tuple[str, float]]:
        """Contribution de chaque feature au logit, en points de logit."""
        if self.logistic.weights is None or self.logistic.scales is None:
            raise RuntimeError("modele non entraine")
        positive_integer("top", top)
        x = feature_matrix(np.asarray(x, dtype=float).reshape(1, -1), columns=len(self.logistic.weights))
        x = self.logistic._apply_residual(x).ravel()
        contrib = (x / self.logistic.scales) * self.logistic.weights
        contrib = contrib * (1 - self.gbdt_weight if self.gbdt is not None else 1.0)
        contrib = contrib / max(self.temperature, 1e-6)
        names = self.feature_names or tuple(f"f{i}" for i in range(len(contrib)))
        pairs = sorted(zip(names, contrib.tolist()), key=lambda kv: -abs(kv[1]))
        return pairs[:top]

    # -------------------------------------------------------------------- I/O

    def save(self, path: str | Path) -> None:
        if self.logistic.weights is None or self.logistic.scales is None:
            raise RuntimeError("modele non entraine")
        path = Path(path)
        payload = {
            "version": 1,
            "l2": self.l2,
            "gbdt_weight": self.gbdt_weight if self.gbdt is not None else 0.0,
            "temperature": self.temperature,
            "feature_names": list(self.feature_names),
            "weights": self.logistic.weights.tolist(),
            "scales": self.logistic.scales.tolist(),
            "resid": self.logistic.resid.tolist() if self.logistic.resid is not None else [],
            "cov": self.logistic.cov.tolist() if self.logistic.cov is not None else None,
            "trained_on": self.trained_on,
            "last_date": self.last_date,
            "has_gbdt": self.gbdt is not None,
        }
        if self.gbdt is not None:
            blob = pickle.dumps(self.gbdt)
            atomic_write_bytes(path.with_suffix(".gbdt.pkl"), blob)
            payload["gbdt_sha256"] = hashlib.sha256(blob).hexdigest()
        atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))

    @classmethod
    def load(cls, path: str | Path) -> "MatchPredictor":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("le fichier de modele doit contenir un objet JSON")
        if type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("version de modele non prise en charge")
        required = {"weights", "scales", "feature_names", "temperature", "l2", "gbdt_weight"}
        if not required <= payload.keys():
            raise ValueError(f"parametres de modele manquants : {sorted(required - payload.keys())}")
        try:
            weights = np.asarray(payload["weights"], dtype=float)
            scales = np.asarray(payload["scales"], dtype=float)
            resid = np.asarray(payload.get("resid") or [], dtype=float)
        except (TypeError, OverflowError) as exc:
            raise ValueError("parametres de modele non numeriques") from exc
        names = payload["feature_names"]
        if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("feature_names doit contenir des noms uniques non vides")
        finite_number("temperature", payload["temperature"], strict=True)
        if (weights.ndim != 1 or len(weights) == 0 or scales.shape != weights.shape
                or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(scales))
                or np.any(scales <= 0) or resid.shape not in ((0,), (len(weights)-1,))
                or not np.all(np.isfinite(resid))
                or len(names) not in (0, len(weights))):
            raise ValueError("parametres de modele invalides")
        has_gbdt = payload.get("has_gbdt", False)
        if not isinstance(has_gbdt, bool):
            raise ValueError("has_gbdt doit etre un booleen")
        trained_on = payload.get("trained_on", 0)
        if type(trained_on) is not int or trained_on < 0:
            raise ValueError("trained_on doit etre un entier positif ou nul")
        last_date = payload.get("last_date", "")
        if not isinstance(last_date, str):
            raise ValueError("last_date doit etre une date ISO ou une chaine vide")
        if last_date:
            dt.date.fromisoformat(last_date)
        m = cls(l2=payload["l2"], use_gbdt=has_gbdt, gbdt_weight=payload["gbdt_weight"])
        m.temperature = payload["temperature"]
        m.feature_names = tuple(names)
        m.trained_on = trained_on
        m.last_date = last_date
        m.logistic = LogisticModel(l2=payload["l2"], feature_names=m.feature_names)
        m.logistic.weights, m.logistic.scales, m.logistic.resid = weights, scales, resid
        if payload.get("cov") is not None:
            try:
                cov = np.asarray(payload["cov"], dtype=float)
            except (TypeError, OverflowError) as exc:
                raise ValueError("matrice de covariance non numerique") from exc
            if cov.shape != (len(weights), len(weights)) or not np.all(np.isfinite(cov)):
                raise ValueError("matrice de covariance invalide")
            # Verifier a l'echelle de la matrice pour tolerer les arrondis de
            # l'inversion de Fisher, sans accepter une variance negative.
            scaled_cov = cov / max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
            if not np.allclose(scaled_cov, scaled_cov.T, rtol=0.0, atol=1e-10):
                raise ValueError("matrice de covariance non symetrique")
            try:
                eigenvalues = np.linalg.eigvalsh(0.5 * (scaled_cov + scaled_cov.T))
            except np.linalg.LinAlgError as exc:
                raise ValueError("matrice de covariance indecomposable") from exc
            if float(eigenvalues.min()) < -1e-10:
                raise ValueError("matrice de covariance non positive")
            m.logistic.cov = cov
        blob = path.with_suffix(".gbdt.pkl")
        if has_gbdt:
            if not blob.is_file():
                raise ValueError(f"composant GBDT manquant : {blob}")
            data = blob.read_bytes()
            if payload.get("gbdt_sha256") and hashlib.sha256(data).hexdigest() != payload["gbdt_sha256"]:
                raise ValueError("composant GBDT incoherent avec le modele JSON")
            m.gbdt = pickle.loads(data)
        else:
            m.gbdt = None
            m.gbdt_weight = 0.0
        return m


def map_predictor(**kwargs) -> MatchPredictor:
    """Configuration map commune au service et aux deux backtests."""
    return MatchPredictor(**{**kwargs, "use_gbdt": False})


def permutation_importance(
    model: MatchPredictor,
    X: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "log_loss",
    repeats: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Importance hors-echantillon par permutation (compatible logistique et GBDT).

    Mesure la degradation moyenne de perte (log-loss ou Brier) lorsqu'on melange
    aleatoirement une colonne : une valeur positive signale un vrai apport predictif.
    """
    from .metrics import brier

    y = binary_labels(y)
    X = feature_matrix(X, rows=len(y))
    positive_integer("repeats", repeats)
    if metric not in ("log_loss", "brier"):
        raise ValueError(f"metrique non geree : {metric!r}")
    loss_fn = log_loss if metric == "log_loss" else brier
    base_loss = loss_fn(y, model.predict_proba(X))
    names = model.feature_names or tuple(f"f{i}" for i in range(X.shape[1]))
    rng = np.random.default_rng(seed)
    X_work = np.array(X, dtype=float, copy=True)
    rows = []
    for j in range(X.shape[1]):
        orig = X_work[:, j].copy()
        deltas = []
        for _ in range(repeats):
            X_work[:, j] = rng.permutation(orig)
            deltas.append(loss_fn(y, model.predict_proba(X_work)) - base_loss)
        X_work[:, j] = orig
        rows.append({
            "feature": names[j],
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
        })
    rows.sort(key=lambda r: -r["mean_delta"])
    return rows


def series_win_probability(p_map: float, best_of: int) -> float:
    """Passe d'une proba par map a une proba de serie (maps supposees i.i.d.).

    Approximation volontairement simple : l'independance entre maps est fausse
    (momentum, lecture d'adversaire), mais l'erreur reste petite devant
    l'incertitude sur p_map lui-meme.
    """
    finite_number("p_map", p_map, maximum=1.0)
    positive_integer("best_of", best_of)
    if best_of not in (1, 3, 5):
        raise ValueError(f"format non gere : bo{best_of}")
    p = min(max(float(p_map), 1e-6), 1 - 1e-6)
    wins_needed = best_of // 2 + 1
    total = 0.0
    for losses in range(wins_needed):
        games = wins_needed + losses
        total += math.comb(games - 1, losses) * (p**wins_needed) * ((1 - p) ** losses)
    return float(total)
