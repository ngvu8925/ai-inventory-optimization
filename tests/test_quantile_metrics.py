import pytest

from src.forecasting.metrics import calc_pinball_loss


def test_pinball_loss_penalizes_underforecast_more_at_high_quantile():
    actual = [10.0]
    forecast = [8.0]
    assert calc_pinball_loss(actual, forecast, 0.9) == pytest.approx(1.8)
    assert calc_pinball_loss(actual, forecast, 0.1) == pytest.approx(0.2)


def test_pinball_loss_rejects_invalid_quantile_and_shape():
    with pytest.raises(ValueError):
        calc_pinball_loss([1.0], [1.0], 1.0)
    with pytest.raises(ValueError):
        calc_pinball_loss([1.0, 2.0], [1.0], 0.5)
