"""Checks for strict input validation in comparison table generation."""

import pandas as pd
import pytest

from src.forecasting.run_chronos_backtest import update_model_comparisons


def test_comparison_requires_unskipped_by_fold_files(tmp_path):
    """A missing expected result is explicit and names the opt-out parameter."""
    with pytest.raises(FileNotFoundError, match="baseline_by_fold.csv") as error:
        update_model_comparisons(tmp_path)

    assert "prefixes_not_run" in str(error.value)


def test_comparison_can_explicitly_skip_unrun_prefixes(tmp_path, capsys):
    """Only the explicitly completed models are included in rebuilt tables."""
    expected = {
        "baseline": ["MovingAverage", "Naive", "SeasonalNaive"],
        "lightgbm": ["LightGBM"],
    }
    for prefix, models in expected.items():
        pd.DataFrame(
            [{"model": model, "fold": 0, "WAPE": 0.5, "MASE": 0.5,
              "MAE": 1.0, "RMSE": 1.0, "Bias": 0.0, "Count": 1}
             for model in models]
        ).to_csv(tmp_path / f"{prefix}_by_fold.csv", index=False)
        pd.DataFrame(
            [{"model": model, "WAPE": 0.5, "MASE": 0.5,
              "MAE": 1.0, "RMSE": 1.0, "Bias": 0.0, "Count": 1}
             for model in models]
        ).to_csv(tmp_path / f"{prefix}_overall.csv", index=False)

    skipped = ["chronos2_history", "chronos2_exogenous", "chronos_bolt_small_history"]
    update_model_comparisons(tmp_path, prefixes_not_run=skipped)

    output = pd.read_csv(tmp_path / "model_comparison_overall.csv")
    assert set(output["model"]) == set(expected["baseline"] + expected["lightgbm"])
    captured = capsys.readouterr().out
    assert "Models included in comparison (4/4)" in captured
