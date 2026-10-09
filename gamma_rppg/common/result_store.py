from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _normalize_key_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, str):
        text = value.strip()
        try:
            return f"{float(text):.12g}"
        except (TypeError, ValueError):
            return text
    return str(value)


class AtomicCSVStore:
    """Small atomic upsert store for resumable CSV experiments.

    Rows are indexed by a stable composite key. Updating a task replaces its former
    row rather than appending a duplicate. Every flush writes to a temporary file and
    atomically replaces the destination.
    """

    def __init__(self, path: Union[str, Path], columns: Sequence[str], key_columns: Sequence[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.key_columns = list(key_columns)
        self.rows: Dict[Tuple[str, ...], Dict] = {}
        self.order: List[Tuple[str, ...]] = []
        self._load()

    def _key(self, row: Dict) -> Tuple[str, ...]:
        return tuple(_normalize_key_value(row.get(c, "")) for c in self.key_columns)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = self._key(row)
                if key not in self.rows:
                    self.order.append(key)
                self.rows[key] = dict(row)

    def get(self, key_values: Dict) -> Optional[Dict]:
        return self.rows.get(self._key(key_values))

    def contains(self, key_values: Dict) -> bool:
        return self._key(key_values) in self.rows

    def upsert(self, row: Dict, flush: bool = True) -> None:
        normalized = {column: row.get(column, "") for column in self.columns}
        key = self._key(normalized)
        if key not in self.rows:
            self.order.append(key)
        self.rows[key] = normalized
        if flush:
            self.flush()

    def remove_where(self, predicate) -> int:
        remove = [key for key in self.order if predicate(self.rows[key])]
        if not remove:
            return 0
        remove_set = set(remove)
        self.order = [key for key in self.order if key not in remove_set]
        for key in remove:
            self.rows.pop(key, None)
        self.flush()
        return len(remove)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for key in self.order:
                writer.writerow({c: self.rows[key].get(c, "") for c in self.columns})
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.rows)
