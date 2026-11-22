"""Lecture CSV commune : ne pas masquer une colonne ou une ligne tronquee."""

from __future__ import annotations

from collections import Counter
import csv
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator, Sequence


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Ecrit atomiquement un fichier binaire (fichier temporaire + fsync + replace)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix=f".{out_path.name}.",
                                         suffix=".tmp", delete=False) as fh:
            temporary = Path(fh.name)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, out_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_csv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence]) -> int:
    """Ecrit atomiquement un CSV UTF-8 avec fin de ligne LF et retourne le nombre de lignes."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=out_path.parent, prefix=f".{out_path.name}.",
                                         suffix=".tmp", delete=False) as fh:
            temporary = Path(fh.name)
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
                count += 1
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, out_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def read_rows(path: str | Path, required: Iterable[str]) -> Iterator[tuple[int, dict[str, str]]]:
    """Produit les lignes et leur numero physique, apres controle de structure.

    Les champs vides explicites sont conserves. Une cellule absente, une cellule
    surnumeraire ou un nom de colonne duplique est une erreur, jamais une valeur
    par defaut ni un ecrasement silencieux.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, strict=True)
        try:
            fields = reader.fieldnames or []
            duplicates = sorted(name for name, count in Counter(fields).items() if count > 1)
            if duplicates:
                raise ValueError(f"colonnes dupliquees dans {path}: {duplicates}")
            missing = set(required) - set(fields)
            if missing:
                raise ValueError(f"colonnes manquantes dans {path}: {sorted(missing)}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"ligne {reader.line_num} invalide dans {path}: "
                                     f"nombre de champs different de l'entete ({len(fields)})")
                yield reader.line_num, row
        except csv.Error as exc:
            raise ValueError(f"CSV invalide dans {path}, ligne {reader.reader.line_num}: {exc}") from exc
