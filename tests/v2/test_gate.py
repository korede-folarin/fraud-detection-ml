"""
Promotion gate rules (D11)
Run: pytest tests/v2
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.promotion.gate import KS_FLOOR, evaluate_gate  # noqa: E402


def model(precision=0.5, cost=1200, ks=0.9, test_cost=2000, test_ks=0.8):
    return {
        "validate": {"Precision": precision, "Cost": cost, "KS": ks},
        "test": {"Precision": precision, "Cost": test_cost, "KS": test_ks},
    }


def test_challenger_better_everywhere_is_promoted():
    result = evaluate_gate(model(), model(precision=0.6, cost=1100, test_cost=1900))
    assert result.promote
    assert result.stage_reached == "test"
    assert all(ok for _, ok, _ in result.checks)


def test_every_rule_must_pass_not_most():
    # better precision and KS, but costs more on validate
    result = evaluate_gate(model(), model(precision=0.9, cost=1300, ks=0.99))
    assert not result.promote
    assert result.outcome == "challenger did not win — keep champion"


def test_test_set_is_not_consulted_when_validate_fails():
    result = evaluate_gate(model(), model(cost=5000, test_cost=0))
    assert result.stage_reached == "validate"
    assert not any(rule.startswith("test") for rule, _, _ in result.checks)


def test_test_confirmation_can_still_block_promotion():
    result = evaluate_gate(model(), model(precision=0.6, cost=1100, test_cost=2500))
    assert not result.promote
    assert result.stage_reached == "test"


def test_ks_floor_applies_on_both_sets():
    assert not evaluate_gate(model(), model(precision=0.6, cost=1100, ks=KS_FLOOR - 0.01)).promote
    assert not evaluate_gate(model(), model(precision=0.6, cost=1100, test_cost=1900, test_ks=KS_FLOOR - 0.01)).promote


def test_ties_count_as_passing():
    assert evaluate_gate(model(), model()).promote


def test_recorded_v1_gate_run_kept_the_champion():
    records = sorted((ROOT / "v2" / "promotion" / "records").glob("*-v1-challengers.json"))
    assert records, "no gate record committed"
    record = json.loads(records[-1].read_text(encoding="utf-8"))
    assert record["champion"] == "XGBoost"
    assert [d["promote"] for d in record["decisions"]] == [False, False]
