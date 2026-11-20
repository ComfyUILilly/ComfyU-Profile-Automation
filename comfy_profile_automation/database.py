from __future__ import annotations

from pathlib import Path

from tinydb import Query, TinyDB


class UsageDatabase:
    """TinyDB-backed usage counters, isolated to a single profile."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "usage.json"
        self.db = TinyDB(self.path, encoding="utf-8")

    def _table(self, profile: str, category: str):
        safe_profile = profile.replace("/", "_")
        return self.db.table(f"{safe_profile}__{category}")

    def record(self, profile: str, checkpoint: str, character: str | None, loras: list[str]) -> None:
        query = Query()
        values_by_category = {
            "checkpoint": [checkpoint],
            "character": [character] if character else [],
            "lora": loras,
        }
        for category, values in values_by_category.items():
            table = self._table(profile, category)
            for value in values:
                row = table.get(query.asset == value)
                if row:
                    table.update({"count": int(row.get("count", 0)) + 1}, doc_ids=[row.doc_id])
                else:
                    table.insert({"asset": value, "count": 1})
        combination = self._table(profile, "combination")
        signature = {"checkpoint": checkpoint, "character": character, "loras": sorted(loras)}
        row = combination.get(query.signature == signature)
        if row:
            combination.update({"count": int(row.get("count", 0)) + 1}, doc_ids=[row.doc_id])
        else:
            combination.insert({"signature": signature, "count": 1})

    def counts(self, profile: str, category: str) -> dict[str, int]:
        return {row["asset"]: int(row.get("count", 0))
                for row in self._table(profile, category).all()}

    def export_xlsx(self, output: Path | None = None) -> Path:
        import pandas as pd
        output = output or self.root / "usage.xlsx"
        rows = []
        for table_name in self.db.tables():
            for row in self.db.table(table_name).all():
                rows.append({"profile_category": table_name, **row})
        pd.DataFrame(rows).to_excel(output, index=False)
        return output

    def close(self) -> None:
        self.db.close()
