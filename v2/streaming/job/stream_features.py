"""
Streaming feature logic — the v1 batch features, one event at a time (D14)

Plain English: v1 computed three features over the whole table at once:
the log of the amount, the hour of day, and how much the last 100 payment
amounts varied. In a stream there is no table — each payment arrives on its
own — so this rebuilds the same three numbers from a running window, and the
parity check proves they match the batch values.

Technical: mirrors v1 DataTransformation.create_features:
  Amount_log          = log1p(Amount)
  Hour                = (Time // 3600) % 24
  rolling_std_amount  = sample std (ddof=1) of the last 100 amounts in
                        time order, current included; 0 when fewer than 2
                        (pandas rolling(100, min_periods=1).std().fillna(0))
The window is global (not per customer) because v1's was: the dataset has no
customer IDs. That forces one ordered stream (parallelism 1) for this step.

What the code does: pure functions and a small window class with no Flink
dependency, so the same code runs in the Flink job and in unit tests.
"""
import math
from collections import deque

WINDOW = 100


class AmountWindow:
    """The last WINDOW amounts, oldest first"""

    def __init__(self, amounts=None):
        self.amounts = deque(amounts or [], maxlen=WINDOW)

    def push(self, amount):
        self.amounts.append(float(amount))
        return self.std()

    def std(self):
        n = len(self.amounts)
        if n < 2:
            return 0.0
        mean = math.fsum(self.amounts) / n
        return math.sqrt(math.fsum((a - mean) ** 2 for a in self.amounts) / (n - 1))


def engineered(amount, time_seconds, rolling_std):
    return {
        "Amount_log": math.log1p(float(amount)),
        "Hour": int((float(time_seconds) // 3600) % 24),
        "rolling_std_amount": rolling_std,
    }


def model_row(event, features, feature_order):
    """Model input in the v1 feature order (final_features.json)"""
    merged = {**event, **features}
    return [float(merged[name]) for name in feature_order]
