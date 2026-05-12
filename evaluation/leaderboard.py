"""
Build, save, and inject the results leaderboard.

The leaderboard averages per-ticker metrics across all tickers and ranks
models by RMSE (lower is better).  Results are saved as:
  - results/leaderboard.csv   → machine-readable
  - results/leaderboard.md    → injected into README.md between markers
"""

import os

import pandas as pd


def build_leaderboard(results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-(model, ticker) results into a summary leaderboard.

    Args:
        results: List of dicts, each with keys:
                 'model', 'ticker', 'MAE', 'RMSE', 'MAPE', 'DA'

    Returns:
        DataFrame with one row per model, metrics averaged across tickers,
        sorted by RMSE ascending.
    """
    df = pd.DataFrame(results)

    summary = (
        df.groupby("model")[["MAE", "RMSE", "MAPE", "DA"]]
        .mean()                      # average across tickers
        .reset_index()
        .sort_values("RMSE")         # best model first
        .reset_index(drop=True)
    )
    # Add a 1-based rank column for readability
    summary.insert(0, "Rank", range(1, len(summary) + 1))
    return summary


def _to_markdown(df: pd.DataFrame) -> str:
    """
    Render a DataFrame as a GitHub-flavored Markdown table.

    Avoids the 'tabulate' dependency so the project has fewer requirements.
    Floats are formatted to 4 decimal places; other values use str().
    """
    cols  = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        cells = []
        for val in row:
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def save_leaderboard(df: pd.DataFrame, out_dir: str = "results") -> None:
    """Write the leaderboard to CSV and Markdown files."""
    os.makedirs(out_dir, exist_ok=True)

    # CSV for downstream analysis or import into other tools
    df.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)

    # Markdown for injection into README.md
    md = _to_markdown(df)
    with open(os.path.join(out_dir, "leaderboard.md"), "w") as f:
        f.write(md)


def inject_into_readme(
    md_path: str = "results/leaderboard.md",
    readme_path: str = "README.md",
) -> None:
    """
    Replace the content between the leaderboard markers in README.md.

    Markers expected in README.md:
        <!-- LEADERBOARD_START -->
        <!-- LEADERBOARD_END -->

    Everything between these two markers is replaced with the contents of
    results/leaderboard.md so the README always shows the latest results.
    """
    start_marker = "<!-- LEADERBOARD_START -->"
    end_marker   = "<!-- LEADERBOARD_END -->"

    with open(readme_path) as f:
        content = f.read()

    # If markers are missing, do nothing rather than corrupting the file
    if start_marker not in content or end_marker not in content:
        return

    with open(md_path) as f:
        table = f.read()

    before = content[: content.index(start_marker) + len(start_marker)]
    after  = content[content.index(end_marker) :]

    with open(readme_path, "w") as f:
        f.write(f"{before}\n\n{table}\n\n{after}")
