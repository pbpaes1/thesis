#!/usr/bin/env python3
"""
Freeze a conservative first-pass state column set (v1) for DRL episodes parquet.

Outputs:
- <output-dir>/v1/allowed_state_columns_v1.json
- <output-dir>/v1/excluded_columns_v1.json
- <output-dir>/v1/state_freeze_v1_summary.md
Notes:
- only one state-freeze markdown summary is produced
- legacy state_freeze_<version>_update_note.md is removed if present
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze allowed/excluded columns for a first-pass DRL state definition."
    )
    parser.add_argument(
        "--input",
        default="/mnt/data/drl_episodes.parquet",
        help="Input parquet file path (default: /mnt/data/drl_episodes.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/freeze",
        help="Freeze root directory where versioned folders will be created (default: data/freeze).",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Freeze version tag, for example v1, v2, v3 (default: v1).",
    )
    parser.add_argument(
        "--max-example-values",
        type=int,
        default=3,
        help="Representative non-null values per column for inspection output.",
    )
    return parser.parse_args()


def example_values(series: pd.Series, max_values: int = 3) -> list[str]:
    values: list[str] = []
    for value in series.dropna():
        text = str(value)
        if text not in values:
            values.append(text)
        if len(values) >= max_values:
            break
    return values


def build_column_profile(df: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column_name": col,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "null_count": int(s.isna().sum()),
                "example_values": " | ".join(example_values(s, max_examples)),
            }
        )
    return pd.DataFrame(rows)


def is_allowed_state_column(col: str) -> bool:
    lower = col.lower()

    if lower.startswith("sma_") or lower.startswith("ema_"):
        return True

    if lower.startswith("rsi_"):
        return True

    if lower in {"macd", "macd_signal", "macd_hist"}:
        return True

    if lower.startswith("bb_"):
        return True

    # Macro/market auxiliary close features (e.g., Gold_Close, VIX_Close).
    if col.endswith("_Close"):
        return True

    if re.fullmatch(r"PC\d+", col):
        return True

    if lower in {
        "unrealized_gains_pct",
        "days_until_tax_transition",
        "unrealized_gain_pct_norm",
        "days_to_tax_transition_norm",
    }:
        return True

    return False


def exclusion_reason(col: str) -> str:
    lower = col.lower()

    if lower == "episode_id" or lower.endswith("_id"):
        return "identifier"

    if lower in {"ticker"}:
        return "identifier excluded in v1 to avoid asset-specific memorization; may be encoded later if needed"

    if lower in {"open", "high", "low", "close", "adj_close", "volume"}:
        return "raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables"

    if lower in {"date", "trigger_date", "tax_transition_date", "simulated_purchase_date"}:
        return "raw temporal metadata; excluded in v1 for conservative state freeze"

    if lower in {"ticker_row", "simulated_purchase_pos"}:
        return "simulator/bookkeeping metadata"

    if lower in {"trigger_candidate", "valid_trigger"}:
        return "leakage risk: episode-construction trigger flag"

    if lower in {
        "simulated_purchase_price",
        "holding_period_days",
    }:
        return "tax/construction variable excluded in v1 to avoid redundant shortcut state"

    if lower == "source":
        return "raw metadata (universe/source provenance), not direct state input in v1"

    return "unclear; conservative exclusion in v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(
    *,
    path: Path,
    source_file: str,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    allowed_columns: list[str],
    excluded_items: list[dict[str, str]],
    version: str,
) -> None:
    lines: list[str] = []
    lines.append(f"# State Freeze {version} Summary")
    lines.append("")
    lines.append("## Dataset overview")
    lines.append("")
    lines.append(f"- Source file: `{source_file}`")
    lines.append(f"- Rows: **{len(df):,}**")
    lines.append(f"- Columns: **{len(df.columns):,}**")
    lines.append("")
    lines.append("Inspection snapshot (name, dtype, examples):")
    lines.append("")
    lines.append("| column_name | dtype | example_values |")
    lines.append("|---|---|---|")
    for row in profile.itertuples(index=False):
        lines.append(
            f"| {row.column_name} | {row.dtype} | {row.example_values} |"
        )
    lines.append("")
    lines.append(f"## Allowed state columns ({version})")
    lines.append("")
    for col in allowed_columns:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append(f"## Excluded columns ({version})")
    lines.append("")
    for item in excluded_items:
        lines.append(f"- `{item['column_name']}`: {item['reason']}")
    lines.append("")
    lines.append("## Rationale (short)")
    lines.append("")
    lines.append("- Included: technical indicators, macro close series, PCA factors, and core tax-aware state (raw + normalized).")
    lines.append("- Tax-aware columns in scope: `unrealized_gains_pct`, `days_until_tax_transition`, `unrealized_gain_pct_norm`, `days_to_tax_transition_norm`.")
    lines.append("- Normalized tax mappings used upstream: `days_to_tax_transition_norm = min(days_until_tax_transition, 365) / 365`, `unrealized_gain_pct_norm = tanh(unrealized_gains_pct / 0.25)`.")
    lines.append("- Excluded: OHLCV, identifiers, dates, trigger flags, simulator bookkeeping, source metadata, and `holding_period_days`.")
    lines.append(f"- This freeze is intentionally conservative and only defines **{version}** state visibility.")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append(f"- This is a state freeze snapshot ({version}), not a final modeling decision.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    version = args.version.strip()
    if not version:
        raise ValueError("Version cannot be empty. Use a tag like v1, v2, or v3.")

    freeze_dir = output_dir / version
    freeze_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input parquet file not found: {input_path}. "
            "Pass a valid path with --input."
        )

    df = pd.read_parquet(input_path)
    profile = build_column_profile(df, max_examples=args.max_example_values)

    print("=== Dataset inspection ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("Columns:")
    print(list(df.columns))
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nExample values by column:")
    print(profile[["column_name", "dtype", "example_values"]].to_string(index=False))

    allowed_columns = [col for col in df.columns if is_allowed_state_column(col)]
    excluded_columns = [col for col in df.columns if col not in allowed_columns]

    excluded_items = [
        {"column_name": col, "reason": exclusion_reason(col)}
        for col in excluded_columns
    ]

    source_file = str(input_path)

    allowed_payload = {
        "version": version,
        "source_file": source_file,
        "allowed_state_columns": allowed_columns,
    }
    excluded_payload = {
        "version": version,
        "source_file": source_file,
        "excluded_columns": excluded_items,
    }

    allowed_path = freeze_dir / f"allowed_state_columns_{version}.json"
    excluded_path = freeze_dir / f"excluded_columns_{version}.json"
    summary_path = freeze_dir / f"state_freeze_{version}_summary.md"
    legacy_update_note_path = freeze_dir / f"state_freeze_{version}_update_note.md"

    write_json(allowed_path, allowed_payload)
    write_json(excluded_path, excluded_payload)
    write_summary(
        path=summary_path,
        source_file=source_file,
        df=df,
        profile=profile,
        allowed_columns=allowed_columns,
        excluded_items=excluded_items,
        version=version,
    )

    if legacy_update_note_path.exists():
        legacy_update_note_path.unlink()
        print(f"Removed legacy file: {legacy_update_note_path}")

    print("\n=== Output files ===")
    print(allowed_path)
    print(excluded_path)
    print(summary_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
