"""Validation tests for EIA subregion ingestion."""

import unittest

from ingestion.energy.fetch_eia_subregion_demand import normalize_rows


def row(period="2025-01-01T00", subba="CE"):
    return {
        "period": period,
        "parent": "PJM",
        "parent-name": "PJM Interconnection, LLC",
        "subba": subba,
        "subba-name": "Commonwealth Edison zone",
        "value": "12000",
        "value-units": "megawatthours",
    }


class NormalizeRowsTests(unittest.TestCase):
    def test_casts_values_and_utc_timestamps(self):
        frame = normalize_rows([row(), row("2025-01-01T01")], "PJM", "CE")
        self.assertEqual(frame.value.tolist(), [12000, 12000])
        self.assertEqual(str(frame.period.dt.tz), "UTC")

    def test_rejects_wrong_or_duplicate_series_data(self):
        with self.assertRaises(ValueError):
            normalize_rows([row(subba="AEP")], "PJM", "CE")
        with self.assertRaises(ValueError):
            normalize_rows([row(), row()], "PJM", "CE")

    def test_rejects_unexpected_units(self):
        bad = row()
        bad["value-units"] = "megawatts"
        with self.assertRaises(ValueError):
            normalize_rows([bad], "PJM", "CE")


if __name__ == "__main__":
    unittest.main()
