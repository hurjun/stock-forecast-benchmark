"""Tests for leaderboard aggregation, ranking, and README injection."""

import pandas as pd
import pytest

from evaluation.leaderboard import (
    build_leaderboard,
    inject_into_readme,
    save_leaderboard,
)


def _sample_results():
    # Two tickers, two models; "good" model has uniformly lower error.
    return [
        {"model": "good", "ticker": "A", "MAE": 1.0, "RMSE": 2.0, "MAPE": 5.0, "DA": 60.0},
        {"model": "good", "ticker": "B", "MAE": 3.0, "RMSE": 4.0, "MAPE": 7.0, "DA": 40.0},
        {"model": "bad",  "ticker": "A", "MAE": 9.0, "RMSE": 10.0, "MAPE": 50.0, "DA": 20.0},
        {"model": "bad",  "ticker": "B", "MAE": 11.0, "RMSE": 12.0, "MAPE": 60.0, "DA": 10.0},
    ]


def test_build_leaderboard_averages_across_tickers():
    lb = build_leaderboard(_sample_results())
    good = lb.set_index("model").loc["good"]
    assert good["RMSE"] == pytest.approx(3.0)   # mean(2, 4)
    assert good["MAE"] == pytest.approx(2.0)     # mean(1, 3)


def test_build_leaderboard_ranks_by_rmse_ascending():
    lb = build_leaderboard(_sample_results())
    assert list(lb["model"]) == ["good", "bad"]
    assert list(lb["Rank"]) == [1, 2]


def test_save_leaderboard_writes_csv_and_markdown(tmp_path):
    lb = build_leaderboard(_sample_results())
    save_leaderboard(lb, out_dir=str(tmp_path))

    csv = tmp_path / "leaderboard.csv"
    md = tmp_path / "leaderboard.md"
    assert csv.exists() and md.exists()

    reloaded = pd.read_csv(csv)
    assert list(reloaded["model"]) == ["good", "bad"]

    table = md.read_text()
    assert table.startswith("| Rank | model |")
    assert "good" in table and "bad" in table


def test_inject_into_readme_replaces_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- LEADERBOARD_START -->\nOLD TABLE\n<!-- LEADERBOARD_END -->\n\nFooter\n"
    )
    md = tmp_path / "leaderboard.md"
    md.write_text("| Rank | model |\n| --- | --- |\n| 1 | good |")

    inject_into_readme(md_path=str(md), readme_path=str(readme))

    out = readme.read_text()
    assert "OLD TABLE" not in out
    assert "| 1 | good |" in out
    assert out.startswith("# Title")
    assert out.rstrip().endswith("Footer")


def test_inject_into_readme_is_idempotent(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- LEADERBOARD_START -->\nplaceholder\n<!-- LEADERBOARD_END -->\n"
    )
    md = tmp_path / "leaderboard.md"
    md.write_text("TABLE_CONTENT")

    inject_into_readme(md_path=str(md), readme_path=str(readme))
    first = readme.read_text()
    inject_into_readme(md_path=str(md), readme_path=str(readme))
    second = readme.read_text()

    assert first == second
    assert second.count("TABLE_CONTENT") == 1


def test_inject_into_readme_noop_without_markers(tmp_path):
    readme = tmp_path / "README.md"
    original = "# No markers here\n"
    readme.write_text(original)
    md = tmp_path / "leaderboard.md"
    md.write_text("TABLE")

    inject_into_readme(md_path=str(md), readme_path=str(readme))
    assert readme.read_text() == original
