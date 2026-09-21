"""Regression checks for temporal alignment and the one-step forecast contract."""

import unittest

import numpy as np
import pandas as pd

from features.energy.engineer import (
    add_lag_and_rolling_features, build_features, clean_outliers, join_weather,
)
from models.energy.train_xgboost import (
    FEATURE_COLUMNS, describe_demand_series, regression_metrics, split_chronologically,
)


def demand_frame(hours=600):
    return pd.DataFrame({
        "period": pd.date_range("2022-12-15", periods=hours, freq="h"),
        "value": 50000.0 + np.arange(hours),
        "respondent": "PJM",
    })


class FeatureTests(unittest.TestCase):
    def test_missing_timestamp_does_not_compress_lags(self):
        raw = demand_frame().drop(index=200)
        features = add_lag_and_rolling_features(clean_outliers(raw, 300000))
        self.assertEqual(len(features), 600)
        self.assertTrue(pd.isna(features.loc[201, "demand_lag_1"]))
        self.assertTrue(pd.isna(features.loc[224, "demand_lag_24"]))
        self.assertEqual(features.loc[225, "demand_lag_24"], 50201)
        self.assertTrue(pd.isna(features.loc[368, "demand_rolling_168"]))
        self.assertEqual(features.loc[369, "demand_lag_168"], 50201)

    def test_invalid_targets_and_history_are_excluded(self):
        raw = demand_frame()
        raw.loc[200, "value"] = 900000
        features = build_features(raw, 300000, include_weather=False)
        times = set(features.period)
        for index in [0, 167, 200, 201, 224, 368]:
            self.assertNotIn(pd.Timestamp(raw.loc[index, "period"], tz="UTC"), times)
        self.assertIn(pd.Timestamp(raw.loc[369, "period"], tz="UTC"), times)

    def test_current_and_future_targets_cannot_change_features_at_t(self):
        raw = demand_frame()
        original = build_features(raw, 300000, include_weather=False)
        changed_raw = raw.copy()
        changed_raw.loc[300:, "value"] += 10000
        changed = build_features(changed_raw, 300000, include_weather=False)
        cutoff = pd.Timestamp(raw.loc[300, "period"], tz="UTC")
        pd.testing.assert_frame_equal(
            original.loc[original.period <= cutoff, FEATURE_COLUMNS],
            changed.loc[changed.period <= cutoff, FEATURE_COLUMNS],
        )
        self.assertNotIn("value", FEATURE_COLUMNS)

    def test_negative_missing_and_infinite_demand_are_masked(self):
        raw = demand_frame()
        raw.loc[:3, "value"] = [-1, np.nan, np.inf, 300000]
        cleaned = clean_outliers(raw, 300000)
        self.assertEqual(len(cleaned), len(raw))
        self.assertTrue(cleaned.loc[:3, "value"].isna().all())
        self.assertEqual(str(cleaned.period.dt.tz), "UTC")

    def test_duplicate_off_hour_and_mixed_series_are_rejected(self):
        raw = demand_frame()
        with self.assertRaises(ValueError):
            clean_outliers(pd.concat([raw, raw.iloc[:1]]), 300000)
        raw.loc[0, "period"] += pd.Timedelta(minutes=30)
        with self.assertRaises(ValueError):
            clean_outliers(raw, 300000)

    def test_subregion_series_is_validated_without_pjm_respondent_column(self):
        raw = demand_frame().drop(columns="respondent")
        raw["parent"] = "PJM"
        raw["subba"] = "CE"
        raw["subba-name"] = "Commonwealth Edison zone"
        raw["value"] = 10000.0 + np.arange(len(raw))
        features = build_features(raw, 50000, include_weather=False)
        self.assertFalse(features.empty)
        description = describe_demand_series(raw)
        self.assertEqual(description["code"], "CE")
        self.assertEqual(description["parent"], "PJM")
        raw.loc[0, "subba"] = "OTHER"
        with self.assertRaises(ValueError):
            clean_outliers(raw, 50000)
        raw = demand_frame()
        raw.loc[0, "respondent"] = "OTHER"
        with self.assertRaises(ValueError):
            clean_outliers(raw, 300000)

    def test_lags_reject_unregularized_input(self):
        with self.assertRaises(ValueError):
            add_lag_and_rolling_features(demand_frame().drop(index=200))

    def test_weather_contract_joins_utc_and_rejects_multiple_locations(self):
        demand = clean_outliers(demand_frame(3), 300000)
        weather = pd.DataFrame({"timestamp_utc": demand.period, "temperature_2m_c": [1, 2, 3]})
        joined = join_weather(demand, weather)
        self.assertEqual(joined.temperature_2m_c.tolist(), [1, 2, 3])
        with self.assertRaises(ValueError):
            join_weather(demand, pd.concat([weather, weather.iloc[:1]]))


class EvaluationTests(unittest.TestCase):
    def test_split_boundaries_are_exclusive_and_disjoint(self):
        frame = pd.DataFrame({"period": pd.to_datetime([
            "2019-12-31", "2020-01-01", "2022-12-31", "2023-01-01",
            "2023-11-30", "2023-12-01", "2023-12-31", "2024-01-01",
        ], utc=True)})
        splits = split_chronologically(frame)
        self.assertEqual([len(value) for value in splits.values()], [2, 2, 2])
        self.assertLess(splits["train"].period.max(), splits["validation"].period.min())
        self.assertLess(splits["validation"].period.max(), splits["test"].period.min())

    def test_empty_or_reversed_splits_fail(self):
        frame = pd.DataFrame({"period": pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")})
        with self.assertRaises(ValueError):
            split_chronologically(frame)
        with self.assertRaises(ValueError):
            split_chronologically(frame, test_start="2022-01-01")

    def test_metrics_use_forecast_errors(self):
        metrics = regression_metrics([10, 20], [13, 16])
        self.assertEqual(metrics["mae"], 3.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(12.5))


if __name__ == "__main__":
    unittest.main()
