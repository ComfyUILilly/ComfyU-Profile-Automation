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
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.worksheet.table import Table, TableStyleInfo
        from openpyxl.utils import get_column_letter
        import json
        import re

        output = output or self.root / "usage.xlsx"
        workbook = Workbook()
        default_sheet = workbook.active
        workbook.remove(default_sheet)
        used_sheet_names = set()
        used_table_names = set()

        def safe_name(value, limit, fallback):
            name = re.sub(r"[^A-Za-z0-9_]", "_", str(value)).strip("_") or fallback
            if not name[0].isalpha() and name[0] != "_":
                name = f"_{name}"
            return name[:limit]

        for index, table_name in enumerate(sorted(self.db.tables()), start=1):
            rows = self.db.table(table_name).all()
            rows = [dict(row) for row in rows]
            for row in rows:
                row.pop("doc_id", None)
                for key, value in list(row.items()):
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value, ensure_ascii=False)
            if rows and all("count" in row for row in rows):
                rows.sort(key=lambda row: int(row.get("count", 0)), reverse=True)

            sheet_base = str(table_name)[:31] or "Usage"
            sheet_name = sheet_base
            suffix = 2
            while sheet_name in used_sheet_names:
                tail = f"_{suffix}"
                sheet_name = sheet_base[:31 - len(tail)] + tail
                suffix += 1
            used_sheet_names.add(sheet_name)
            sheet = workbook.create_sheet(sheet_name)

            headers = list(dict.fromkeys(key for row in rows for key in row))
            if not headers:
                headers = ["No data"]
                sheet.append(headers)
                continue
            sheet.append(headers)
            for row in rows:
                sheet.append([row.get(key, "") for key in headers])
            for cell in sheet[1]:
                cell.font = Font(bold=True)

            if rows:
                table_name_safe = safe_name(table_name, 200, "Usage")
                display_name = table_name_safe
                suffix = 2
                while display_name in used_table_names:
                    display_name = f"{table_name_safe[:190]}_{suffix}"
                    suffix += 1
                used_table_names.add(display_name)
                ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
                excel_table = Table(displayName=display_name, ref=ref)
                excel_table.tableStyleInfo = TableStyleInfo(
                    name="TableStyleMedium9",
                    showFirstColumn=False,
                    showLastColumn=False,
                    showRowStripes=True,
                    showColumnStripes=False,
                )
                sheet.add_table(excel_table)

            for column_index, header in enumerate(headers, start=1):
                letter = get_column_letter(column_index)
                values = [len(str(header)), *[len(str(row.get(header, ""))) for row in rows]]
                sheet.column_dimensions[letter].width = min(202, max(values, default=10) + 2)

        if not workbook.sheetnames:
            workbook.create_sheet("Usage")
        workbook.save(output)
        return output

    def close(self) -> None:
        self.db.close()
