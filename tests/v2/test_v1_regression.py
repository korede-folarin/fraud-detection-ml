"""
v1 must stay v1 (D8, D9)
Run: pytest tests/v2

Runs the regression check in-process so the code it exercises (v1's
ModelEvaluation) is counted in coverage, and checks that tampering is caught.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.regression import check_v1  # noqa: E402


def test_v1_results_reproduce_exactly(capsys):
    check_v1.main()
    assert "v1 regression check passed" in capsys.readouterr().out


def test_a_changed_v1_number_is_reported(monkeypatch, tmp_path):
    tampered = json.loads(check_v1.RESULTS.read_text(encoding="utf-8"))
    tampered["test"]["XGBoost"]["KS"] = 0.999
    fake = tmp_path / "evaluation_results.json"
    fake.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(check_v1, "RESULTS", fake)

    with pytest.raises(SystemExit) as exit_info:
        check_v1.main()
    assert exit_info.value.code == 1
