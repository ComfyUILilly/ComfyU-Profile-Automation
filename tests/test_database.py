import tempfile
import unittest
from pathlib import Path

from tinydb import TinyDB

from comfy_profile_automation.database import UsageDatabase


class UsageDatabaseTests(unittest.TestCase):
    def test_migrates_legacy_count_db_for_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            data_root.mkdir()
            legacy_path = data_root / "count.db"
            legacy = TinyDB(legacy_path, encoding="utf-8")
            legacy.table("IL-Checkpoint").insert({"Checkpoint": "base.safetensors", "count": 7})
            legacy.table("IL-Char").insert({"Char": "hero.safetensors", "count": 3})
            legacy.table("IL-Lora").insert({"Lora": "style.safetensors", "count": 4})
            legacy.table("IL-Combination").insert({
                "Checkpoint": "base.safetensors",
                "Char": "hero.safetensors",
                "Loras": ["style.safetensors"],
                "count": 2,
            })
            legacy.table("IL-Tag").insert({"Tag": "style", "count": 5})
            legacy.table("Anime-Checkpoint").insert({"Checkpoint": "anime.safetensors", "count": 9})
            legacy.close()

            db = UsageDatabase(data_root / "IL" / "database")
            self.assertEqual(db.counts("IL", "checkpoint"), {"base.safetensors": 7})
            self.assertEqual(db.counts("IL", "character"), {"hero.safetensors": 3})
            self.assertEqual(db.counts("IL", "lora"), {"style.safetensors": 4})
            combination = db._table("IL", "combination").all()
            self.assertEqual(combination[0]["signature"]["checkpoint"], "base.safetensors")
            self.assertEqual(combination[0]["count"], 2)
            self.assertEqual(db._table("IL", "tag").all()[0]["tag"], "style")
            self.assertEqual(db._table("Anime", "checkpoint").all(), [])
            db.close()

            reopened = UsageDatabase(data_root / "IL" / "database")
            self.assertEqual(reopened.counts("IL", "checkpoint"), {"base.safetensors": 7})
            self.assertEqual(len(reopened._table("IL", "combination").all()), 1)
            reopened.close()


    def test_xlsx_export_creates_separate_sheets_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = UsageDatabase(Path(tmp) / "IL" / "database")
            db.record("IL", "base.safetensors", "hero.safetensors", ["style.safetensors"])
            db.record("IL", "base.safetensors", "hero.safetensors", ["style.safetensors"])
            output = db.export_xlsx()
            self.assertTrue(output.exists())
            from openpyxl import load_workbook
            workbook = load_workbook(output)
            self.assertIn("IL__checkpoint", workbook.sheetnames)
            sheet = workbook["IL__checkpoint"]
            self.assertEqual(sheet.cell(1, 1).value, "asset")
            self.assertEqual(sheet.cell(2, 2).value, 2)
            self.assertEqual(len(sheet.tables), 1)
            workbook.close()
            db.close()

if __name__ == "__main__":
    unittest.main()
