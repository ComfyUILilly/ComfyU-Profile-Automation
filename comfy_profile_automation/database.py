from __future__ import annotations

from pathlib import Path

from tinydb import Query, TinyDB


class UsageDatabase:
    """TinyDB-backed usage counters, isolated to a single profile."""

    LEGACY_TABLE_SUFFIXES = {
        "checkpoint": "Checkpoint",
        "character": "Char",
        "lora": "Lora",
        "combination": "Combination",
        "tag": "Tag",
    }

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "usage.json"
        self.db = TinyDB(self.path, encoding="utf-8")
        self._migrate_legacy(root.parent.parent / "count.db", root.parent.name)

    def _migrate_legacy(self, legacy_path: Path, profile: str) -> None:
        """Import this profile's historical count.db tables once, without modifying it."""
        if not legacy_path.exists():
            return
        prefix = profile + "-"
        try:
            legacy = TinyDB(legacy_path, encoding="utf-8")
            try:
                for category, suffix in self.LEGACY_TABLE_SUFFIXES.items():
                    legacy_name = prefix + suffix
                    if legacy_name not in legacy.tables():
                        continue
                    target = self._table(profile, category)
                    if target.all():
                        continue
                    for row in legacy.table(legacy_name).all():
                        if category in ("checkpoint", "character", "lora"):
                            asset = row.get({"checkpoint": "Checkpoint", "character": "Char", "lora": "Lora"}[category])
                            if asset:
                                target.insert({"asset": asset, "count": int(row.get("count", 0))})
                        elif category == "combination":
                            checkpoint = row.get("Checkpoint")
                            character = row.get("Char")
                            loras = row.get("Loras", [])
                            if checkpoint is not None:
                                target.insert({
                                    "signature": {
                                        "checkpoint": checkpoint,
                                        "character": character,
                                        "loras": sorted(loras) if isinstance(loras, list) else [],
                                    },
                                    "count": int(row.get("count", 0)),
                                })
                        elif category == "tag":
                            tag = row.get("Tag")
                            if tag:
                                target.insert({"tag": tag, "count": int(row.get("count", 0))})
            finally:
                legacy.close()
        except Exception:
            # Keep the automation usable; expose migration failure in the logs.
            import logging
            logging.getLogger(__name__).exception("Legacy usage database migration failed: %s", legacy_path)

    def _table(self, profile: str, category: str):
        safe_profile = profile.replace("/", "_")
        return self.db.table(f"{safe_profile}__{category}")

    def record(self, profile: str, checkpoint: str, character: str | None, loras: list[str],
               tags: list[str] | None = None) -> None:
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
        tag_table = self._table(profile, "tag")
        for tag in tags or []:
            tag = str(tag).strip()
            if not tag:
                continue
            row = tag_table.get(query.tag == tag)
            if row:
                tag_table.update({"count": int(row.get("count", 0)) + 1}, doc_ids=[row.doc_id])
            else:
                tag_table.insert({"tag": tag, "count": 1})

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
