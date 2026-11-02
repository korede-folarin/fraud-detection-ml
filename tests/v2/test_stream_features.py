"""
Streaming features must equal v1 batch features (D14)
Run: pytest tests/v2

The raw transactions are not in the repo (DVC, authenticated), so this checks
the window maths against pandas rolling std (v1's method) on a synthetic
heavy-tailed stream with zeros and repeats, plus the Hour / Amount_log formulas.
The full check on the real 42,722 test rows is the streaming parity run
(v2/streaming/parity.py), whose result is committed in v2/streaming/results/.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v2" / "streaming" / "job"))

from stream_features import WINDOW, AmountWindow, engineered, model_row  # noqa: E402


def test_running_window_matches_pandas_rolling_std():
    rng = np.random.default_rng(7)
    # heavy-tailed amounts with exact repeats and zeros, like the card data
    amounts = np.round(np.exp(rng.normal(3, 1.6, 2500)), 2)
    amounts[::17] = 0.0
    amounts[5::23] = amounts[4::23][: len(amounts[5::23])]

    expected = pd.Series(amounts).rolling(WINDOW, min_periods=1).std().fillna(0).to_numpy()
    window = AmountWindow()
    got = np.array([window.push(a) for a in amounts])

    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_first_event_has_zero_volatility():
    assert AmountWindow().push(42.0) == 0.0


def test_window_keeps_only_the_last_100():
    window = AmountWindow(range(WINDOW))
    window.push(1000)
    assert len(window.amounts) == WINDOW and window.amounts[0] == 1


def test_hour_and_amount_log_follow_v1_definitions():
    f = engineered(amount=99.0, time_seconds=26 * 3600 + 59, rolling_std=1.5)
    assert f["Hour"] == 2                     # hour 26 of the window is 02:xx on day two
    assert f["Amount_log"] == math.log1p(99.0)
    assert f["rolling_std_amount"] == 1.5


def test_model_row_uses_v1_feature_order():
    order = ["V14", "Amount_log", "Hour", "rolling_std_amount"]
    event = {"V14": -3.0, "Amount": 9.0, "Time": 0}
    row = model_row(event, {"Amount_log": 2.3, "Hour": 0, "rolling_std_amount": 0.0}, order)
    assert row == [-3.0, 2.3, 0.0, 0.0]
