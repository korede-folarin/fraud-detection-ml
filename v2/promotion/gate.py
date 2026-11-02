"""
Promotion gate — the only way a challenger can replace the champion (D11)

Plain English: a new model only replaces the current one if it is at least as
good where it matters to the bank — at a real decision threshold, on data it
was not trained on — and it must pass every rule, not most of them.

Technical: a pure function over two metric dicts. Selection uses the validate
set only (each model at its own validate cost-optimal threshold). The test set
is consulted once, as a confirmation, and only if validate passed — so test
data never influences which model is chosen.

What the code does: returns a GateResult with the decision ("promote" or
"keep champion") and one line per rule saying which passed and which failed.
"""
from dataclasses import dataclass, field

KS_FLOOR = 0.75   # same line as the Airflow drift_check retraining trigger


@dataclass(frozen=True)
class GateResult:
    promote: bool
    stage_reached: str           # "validate" or "test"
    checks: list = field(default_factory=list)   # (rule, passed, detail)

    @property
    def outcome(self):
        return "promote challenger" if self.promote else "challenger did not win — keep champion"

    def summary(self):
        lines = [f"Outcome: {self.outcome} (stopped at {self.stage_reached})"]
        lines += [f"  [{'PASS' if ok else 'FAIL'}] {rule}: {detail}" for rule, ok, detail in self.checks]
        return "\n".join(lines)


def evaluate_gate(champion, challenger):
    """
    champion / challenger: {"validate": {...}, "test": {...}}, each split with
    Precision, Cost, KS measured at that model's own validate-chosen threshold
    """
    checks = []
    cv, hv = champion["validate"], challenger["validate"]
    checks.append(("validate alert precision >= champion",
                   hv["Precision"] >= cv["Precision"], f"{hv['Precision']:.4f} vs {cv['Precision']:.4f}"))
    checks.append(("validate business cost <= champion",
                   hv["Cost"] <= cv["Cost"], f"£{hv['Cost']:,.0f} vs £{cv['Cost']:,.0f}"))
    checks.append((f"validate KS >= {KS_FLOOR}",
                   hv["KS"] >= KS_FLOOR, f"{hv['KS']:.4f}"))

    if not all(ok for _, ok, _ in checks):
        return GateResult(promote=False, stage_reached="validate", checks=checks)

    # one confirmation on test, frozen thresholds, no re-tuning
    ct, ht = champion["test"], challenger["test"]
    checks.append(("test business cost <= champion",
                   ht["Cost"] <= ct["Cost"], f"£{ht['Cost']:,.0f} vs £{ct['Cost']:,.0f}"))
    checks.append((f"test KS >= {KS_FLOOR}",
                   ht["KS"] >= KS_FLOOR, f"{ht['KS']:.4f}"))

    return GateResult(promote=all(ok for _, ok, _ in checks), stage_reached="test", checks=checks)
