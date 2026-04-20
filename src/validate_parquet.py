#!/usr/bin/env python3
"""
Validate an episode-level parquet dataset for DRL tax-aware stopping tasks.

Outputs:
- parquet_data_dictionary.csv
- parquet_column_classification.csv
- parquet_validation_report.md
- duplicate_episode_date_rows.csv
- episode_date_order_issues.csv
- holding_period_issues.csv
- critical_missingness_summary.csv
- trigger_date_issues.csv
- tax_transition_issues.csv
- unrealized_gain_diagnostics.csv
- unrealized_gain_formula_scores.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DICTIONARY_FILE = "parquet_data_dictionary.csv"
COLUMN_CLASSIFICATION_FILE = "parquet_column_classification.csv"
VALIDATION_REPORT_FILE = "parquet_validation_report.md"

DUPLICATE_EPISODE_DATE_FILE = "duplicate_episode_date_rows.csv"
EPISODE_ORDER_ISSUES_FILE = "episode_date_order_issues.csv"
HOLDING_PERIOD_ISSUES_FILE = "holding_period_issues.csv"
CRITICAL_MISSINGNESS_FILE = "critical_missingness_summary.csv"
TRIGGER_DATE_ISSUES_FILE = "trigger_date_issues.csv"
TAX_TRANSITION_ISSUES_FILE = "tax_transition_issues.csv"
UNREALIZED_GAIN_DIAGNOSTICS_FILE = "unrealized_gain_diagnostics.csv"
UNREALIZED_GAIN_SCORES_FILE = "unrealized_gain_formula_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate parquet episode dataset and produce validation artifacts."
    )
    parser.add_argument(
        "--input",
        default="/mnt/data/drl_episodes.parquet",
        help="Path to input parquet file (default: /mnt/data/drl_episodes.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where outputs will be written (default: current directory).",
    )
    parser.add_argument(
        "--max-infer-sample",
        type=int,
        default=10000,
        help="Max non-null rows sampled per column for type inference.",
    )
    parser.add_argument(
        "--max-example-values",
        type=int,
        default=5,
        help="Max representative values per column in data dictionary.",
    )
    parser.add_argument(
        "--diagnostic-sample-size",
        type=int,
        default=500,
        help="Max rows to store in unrealized gain diagnostics sample.",
    )
    return parser.parse_args()


def sample_non_null(series: pd.Series, max_size: int) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) <= max_size:
        return non_null
    step = max(len(non_null) // max_size, 1)
    return non_null.iloc[::step].iloc[:max_size]


def stringify_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        if math.isfinite(float(value)):
            return f"{float(value):.6g}"
        return str(value)
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def representative_values(series: pd.Series, max_values: int) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna():
        text = stringify_value(value)
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
        if len(values) >= max_values:
            break
    return " | ".join(values)


def infer_date_columns(df: pd.DataFrame, sample_size: int) -> dict[str, dict[str, Any]]:
    date_info: dict[str, dict[str, Any]] = {}
    name_tokens = ("date", "time", "timestamp", "datetime")

    for col in df.columns:
        series = df[col]
        lower = col.lower()
        reasons: list[str] = []
        score = 0.0
        parse_ratio = np.nan

        name_hint = any(token in lower for token in name_tokens)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        if name_hint:
            score += 1.0
            reasons.append("name suggests datetime semantics")

        if is_datetime:
            score += 1.5
            reasons.append("dtype is datetime")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sampled = sample_non_null(series, sample_size)
            if len(sampled) > 0:
                # Avoid aggressive parsing for obvious non-date text columns.
                text_sample = sampled.astype(str).head(50)
                date_pattern_ratio = float(
                    text_sample.str.contains(
                        r"(?:\d{4}-\d{2}-\d{2})|(?:\d{1,2}/\d{1,2}/\d{2,4})|(?:\d{1,2}-\d{1,2}-\d{2,4})",
                        regex=True,
                    ).mean()
                )
                should_attempt_parse = name_hint or date_pattern_ratio >= 0.60
                if lower.endswith("_id") or lower == "episode_id":
                    should_attempt_parse = False
                reasons.append(f"date-pattern ratio on sample: {date_pattern_ratio:.3f}")

                if should_attempt_parse:
                    parsed = pd.to_datetime(sampled, errors="coerce")
                    parse_ratio = float(parsed.notna().mean())
                    reasons.append(f"datetime parse ratio on sample: {parse_ratio:.3f}")
                    if parse_ratio >= 0.95:
                        score += 1.2
                    elif parse_ratio >= 0.80:
                        score += 0.8
                    elif parse_ratio >= 0.60:
                        score += 0.4

        is_likely_date = score >= 1.2
        date_info[col] = {
            "is_likely_date": is_likely_date,
            "score": score,
            "parse_ratio": parse_ratio,
            "reasons": "; ".join(reasons) if reasons else "no strong datetime signal",
        }

    return date_info


def convert_date_columns(
    df: pd.DataFrame,
    date_info: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    out = df.copy()
    converted: list[str] = []
    notes: dict[str, str] = {}

    for col, info in date_info.items():
        if not info.get("is_likely_date", False):
            continue

        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            converted.append(col)
            notes[col] = "already datetime dtype"
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        non_null = int(series.notna().sum())
        parse_success = int(parsed.notna().sum())
        parse_ratio = parse_success / non_null if non_null > 0 else 0.0

        if parse_ratio >= 0.70:
            out[col] = parsed
            converted.append(col)
            notes[col] = f"converted to datetime (parse ratio={parse_ratio:.3f})"
        else:
            notes[col] = (
                "kept as original dtype "
                f"(parse ratio too low: {parse_ratio:.3f})"
            )

    return out, converted, notes


def infer_numeric_columns(df: pd.DataFrame, sample_size: int) -> tuple[list[str], list[str]]:
    strict_numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    numeric_like: list[str] = []

    for col in df.columns:
        if col in strict_numeric:
            continue
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        sampled = sample_non_null(series, sample_size)
        if len(sampled) == 0:
            continue
        parsed = pd.to_numeric(sampled, errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio >= 0.95:
            numeric_like.append(col)

    return strict_numeric, numeric_like


def choose_main_date_column(df: pd.DataFrame, date_cols: list[str]) -> tuple[str | None, str]:
    if not date_cols:
        return None, "No date-like column detected."

    scored: list[tuple[float, float, str, str]] = []
    for col in date_cols:
        lower = col.lower()
        series = df[col]
        non_null_ratio = float(series.notna().mean()) if len(series) else 0.0

        score = 0.0
        reasons: list[str] = []
        if lower == "date":
            score += 100.0
            reasons.append("exact name match 'date'")
        if "date" in lower:
            score += 20.0
            reasons.append("contains 'date'")
        if any(token in lower for token in ("trigger", "transition", "purchase")):
            score -= 30.0
            reasons.append("looks like event date rather than per-step date")
        if "tax" in lower:
            score -= 10.0
            reasons.append("tax-event naming signal")
        if pd.api.types.is_datetime64_any_dtype(series):
            score += 5.0
            reasons.append("datetime dtype")
        score += non_null_ratio * 5.0
        reasons.append(f"non-null ratio={non_null_ratio:.3f}")

        scored.append((score, non_null_ratio, col, "; ".join(reasons)))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_score, _, best_col, best_reason = scored[0]
    explanation = (
        f"Selected '{best_col}' as the main time-step column "
        f"(score={best_score:.2f}; {best_reason})."
    )
    return best_col, explanation


def infer_meaning_units_role(
    column: str,
    series: pd.Series,
    date_cols: set[str],
) -> tuple[str, str, str]:
    lower = column.lower()

    if column in date_cols:
        if "trigger" in lower:
            return (
                "Likely trigger event date for the episode.",
                "Datetime.",
                "Likely tax/event timing variable.",
            )
        if "transition" in lower:
            return (
                "Likely tax-regime transition date (for example LTCG threshold).",
                "Datetime.",
                "Likely tax/event timing variable.",
            )
        if "purchase" in lower:
            return (
                "Likely simulated purchase date associated with cost basis.",
                "Datetime.",
                "Likely simulator/tax metadata.",
            )
        return (
            "Likely per-step timestamp.",
            "Datetime.",
            "Likely temporal index.",
        )

    if lower == "episode_id" or lower.endswith("_id"):
        return (
            "Likely unique episode identifier.",
            "N/A (identifier).",
            "Identifier.",
        )

    if lower == "ticker":
        return (
            "Likely stock ticker symbol.",
            "N/A (symbol).",
            "Identifier / grouping key.",
        )

    if lower in {"open", "high", "low", "close"}:
        return (
            f"Likely {lower} price for the instrument.",
            "Likely currency units (often USD).",
            "Likely observation feature.",
        )

    if lower == "adj_close":
        return (
            "Likely adjusted close price (split/dividend adjusted).",
            "Likely currency units (often USD).",
            "Likely observation feature.",
        )

    if lower.endswith("_close"):
        return (
            "Likely close/level of an auxiliary market or macro series.",
            "Likely index level or price level; confirm source convention.",
            "Likely observation feature.",
        )

    if lower == "volume":
        return (
            "Likely traded volume.",
            "Likely shares/contracts traded.",
            "Likely observation feature.",
        )

    if lower.startswith("sma_") or lower.startswith("ema_"):
        return (
            "Likely moving-average technical indicator.",
            "Unitless after scaling, or price-like before scaling; confirm preprocessing.",
            "Likely observation feature.",
        )

    if lower.startswith("pc") and lower[2:].isdigit():
        return (
            "Likely rolling PCA factor score.",
            "Unitless standardized component score.",
            "Likely observation feature.",
        )

    if lower.startswith("rsi"):
        return (
            "Likely RSI technical indicator.",
            "RSI scale (typically 0-100) or standardized variant.",
            "Likely observation feature.",
        )

    if "macd" in lower:
        return (
            "Likely MACD-based technical indicator.",
            "Indicator units (often price-difference scale or standardized).",
            "Likely observation feature.",
        )

    if lower.startswith("bb_"):
        return (
            "Likely Bollinger Bands-derived indicator.",
            "Indicator scale (price-like or standardized).",
            "Likely observation feature.",
        )

    if "holding_period" in lower and "day" in lower:
        return (
            "Likely elapsed holding period since simulated purchase.",
            "Days (likely calendar days).",
            "Likely tax-aware state variable.",
        )

    if "unrealized" in lower and "gain" in lower:
        return (
            "Likely unrealized gain relative to purchase cost basis.",
            "Fraction or percentage; needs manual confirmation.",
            "Likely tax-aware state variable.",
        )

    if "purchase_price" in lower:
        return (
            "Likely simulated purchase/cost basis price.",
            "Likely currency units (often USD).",
            "Likely tax-aware state variable.",
        )

    if "purchase" in lower and "pos" in lower:
        return (
            "Likely positional index of simulated purchase row in source sequence.",
            "Row index (integer).",
            "Likely simulator-only metadata.",
        )

    if lower in {"valid_trigger", "trigger_candidate"}:
        return (
            "Likely internal trigger flag used in simulator/episode construction.",
            "Boolean flag.",
            "Likely simulator-only or leakage-prone field.",
        )

    if lower == "ticker_row":
        return (
            "Likely per-ticker row index in preprocessing pipeline.",
            "Row index (integer).",
            "Likely simulator-only metadata.",
        )

    if lower == "source":
        return (
            "Likely source/universe tag for instrument origin.",
            "Categorical label.",
            "Likely simulator/universe metadata; often excluded from observations.",
        )

    # Fallback
    if pd.api.types.is_numeric_dtype(series):
        return (
            "Numeric field with unclear semantics from name alone.",
            "Unclear / needs manual review.",
            "Unclear / needs manual review.",
        )

    return (
        "Unclear / needs manual review.",
        "Unclear / needs manual review.",
        "Unclear / needs manual review.",
    )


def classify_usage(
    column: str,
    series: pd.Series,
    date_cols: set[str],
    main_date_col: str | None,
) -> tuple[str, str]:
    lower = column.lower()

    if lower == "episode_id" or lower.endswith("_id") or lower == "ticker":
        return "identifier", "Identifier field used to group rows into episodes/instruments."

    if main_date_col is not None and column == main_date_col:
        return "identifier", "Primary per-step time index."

    if column in date_cols and any(t in lower for t in ("trigger", "transition", "purchase")):
        return "tax_variable", "Episode event timing linked to tax logic or simulated purchase."

    if lower in {
        "holding_period_days",
        "unrealized_gains_pct",
        "simulated_purchase_price",
        "simulated_purchase_date",
        "trigger_date",
        "tax_transition_date",
    }:
        return "tax_variable", "Likely tax-aware state/cost-basis variable."

    if lower in {"ticker_row", "simulated_purchase_pos"}:
        return "simulator_only", "Implementation bookkeeping field from simulator/preprocessing."

    if lower in {"trigger_candidate", "valid_trigger"}:
        return (
            "excluded",
            "Trigger-construction flag likely not directly observable at decision time; leakage risk.",
        )

    if lower == "source":
        return (
            "excluded",
            "Universe/source provenance metadata; excluded by policy choice to avoid dataset-construction leakage.",
        )

    if lower in {"open", "high", "low", "close", "adj_close", "volume"}:
        return "observation_feature", "Raw market state variable."

    if lower.endswith("_close"):
        return "observation_feature", "Auxiliary market/macro close level feature."

    if lower.startswith(("sma_", "ema_", "rsi", "macd", "bb_")):
        return "observation_feature", "Technical indicator feature."

    if lower.startswith("pc") and lower[2:].isdigit():
        return "observation_feature", "PCA factor feature."

    if pd.api.types.is_numeric_dtype(series):
        return (
            "observation_feature",
            "Numeric feature with uncertain semantics; assigned as observation by best effort.",
        )

    if column in date_cols:
        return "identifier", "Datetime field used as temporal reference."

    return (
        "excluded",
        "Semantics unclear or likely non-observable metadata; exclude until manually confirmed.",
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_data_dictionary(
    df: pd.DataFrame,
    date_cols: list[str],
    max_example_values: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    date_set = set(date_cols)

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        null_pct = float((null_count / len(series) * 100.0) if len(series) else 0.0)
        unique_count = int(series.nunique(dropna=True))
        examples = representative_values(series, max_example_values)
        meaning, units, role = infer_meaning_units_role(col, series, date_set)

        rows.append(
            {
                "column_name": col,
                "dtype": str(series.dtype),
                "non_null_count": non_null,
                "null_count": null_count,
                "null_pct": round(null_pct, 6),
                "unique_count": unique_count,
                "example_values": examples,
                "inferred_meaning": meaning,
                "inferred_units": units,
                "inferred_role": role,
            }
        )

    return pd.DataFrame(rows)


def build_column_classification(
    df: pd.DataFrame,
    date_cols: list[str],
    main_date_col: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    date_set = set(date_cols)

    for col in df.columns:
        usage_class, rationale = classify_usage(col, df[col], date_set, main_date_col)
        rows.append(
            {
                "column_name": col,
                "usage_class": usage_class,
                "rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


def check_duplicate_episode_date(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    out_path = output_dir / DUPLICATE_EPISODE_DATE_FILE
    if episode_col is None or date_col is None:
        empty = pd.DataFrame(
            columns=["episode_id", "date", "reason"]
        )
        write_csv(empty, out_path)
        return {
            "name": "duplicate_episode_date",
            "status": "skipped",
            "message": "episode_id and/or main date column not available.",
            "output_file": out_path.name,
            "duplicate_rows": 0,
            "duplicate_keys": 0,
        }

    dup_mask = df.duplicated(subset=[episode_col, date_col], keep=False)
    dup_rows = df.loc[dup_mask].copy()

    if not dup_rows.empty:
        dup_rows = dup_rows.sort_values([episode_col, date_col], kind="stable")
    write_csv(dup_rows, out_path)

    duplicate_rows = int(len(dup_rows))
    duplicate_keys = int(
        dup_rows[[episode_col, date_col]].drop_duplicates().shape[0]
        if duplicate_rows > 0
        else 0
    )
    status = "pass" if duplicate_rows == 0 else "fail"
    message = (
        "No duplicate (episode_id, date) rows found."
        if status == "pass"
        else f"Found {duplicate_rows} duplicate rows across {duplicate_keys} duplicate keys."
    )
    return {
        "name": "duplicate_episode_date",
        "status": status,
        "message": message,
        "output_file": out_path.name,
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
    }


def check_episode_date_order(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    out_path = output_dir / EPISODE_ORDER_ISSUES_FILE
    if episode_col is None or date_col is None:
        empty = pd.DataFrame(
            columns=["episode_id", "row_number", "previous_date", "current_date", "issue_type"]
        )
        write_csv(empty, out_path)
        return {
            "name": "episode_date_order",
            "status": "skipped",
            "message": "episode_id and/or main date column not available.",
            "output_file": out_path.name,
            "issue_rows": 0,
            "episodes_with_issues": 0,
        }

    tmp = pd.DataFrame(
        {
            episode_col: df[episode_col],
            date_col: df[date_col],
            "row_number": np.arange(len(df), dtype=np.int64),
        }
    )
    tmp["previous_date"] = tmp.groupby(episode_col, sort=False)[date_col].shift(1)
    issue_mask = (
        tmp[date_col].notna()
        & tmp["previous_date"].notna()
        & (tmp[date_col] < tmp["previous_date"])
    )
    issues = tmp.loc[issue_mask, [episode_col, "row_number", "previous_date", date_col]].copy()
    issues = issues.rename(columns={episode_col: "episode_id", date_col: "current_date"})
    if not issues.empty:
        issues["issue_type"] = "date_not_monotonic_in_input_order"
        issues = issues.sort_values(["episode_id", "row_number"], kind="stable")
    else:
        issues["issue_type"] = []

    write_csv(issues, out_path)

    issue_rows = int(len(issues))
    episodes_with_issues = int(issues["episode_id"].nunique()) if issue_rows > 0 else 0
    status = "pass" if issue_rows == 0 else "fail"
    message = (
        "All episodes have non-decreasing dates in dataset row order."
        if status == "pass"
        else f"Found {issue_rows} ordering issue rows across {episodes_with_issues} episodes."
    )
    return {
        "name": "episode_date_order",
        "status": status,
        "message": message,
        "output_file": out_path.name,
        "issue_rows": issue_rows,
        "episodes_with_issues": episodes_with_issues,
    }


def detect_holding_period_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        lower = col.lower()
        is_candidate = (
            lower == "holding_period_days"
            or ("holding" in lower and "day" in lower)
            or (lower.endswith("_days") and "holding" in lower)
        )
        if is_candidate and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    return candidates


def check_holding_period(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    out_path = output_dir / HOLDING_PERIOD_ISSUES_FILE
    if episode_col is None:
        empty = pd.DataFrame(
            columns=[
                "episode_id",
                "row_number",
                "column_name",
                "date",
                "previous_date",
                "value",
                "previous_value",
                "delta_value",
                "delta_date_days",
                "issue_type",
            ]
        )
        write_csv(empty, out_path)
        return {
            "name": "holding_period_monotonicity",
            "status": "skipped",
            "message": "episode_id column not available.",
            "output_file": out_path.name,
            "issue_rows": 0,
            "episodes_with_issues": 0,
            "columns_checked": [],
        }

    holding_cols = detect_holding_period_columns(df)
    if not holding_cols:
        empty = pd.DataFrame(
            columns=[
                "episode_id",
                "row_number",
                "column_name",
                "date",
                "previous_date",
                "value",
                "previous_value",
                "delta_value",
                "delta_date_days",
                "issue_type",
            ]
        )
        write_csv(empty, out_path)
        return {
            "name": "holding_period_monotonicity",
            "status": "skipped",
            "message": "No likely holding-period numeric column found.",
            "output_file": out_path.name,
            "issue_rows": 0,
            "episodes_with_issues": 0,
            "columns_checked": [],
        }

    issues_all: list[pd.DataFrame] = []
    for col in holding_cols:
        work = pd.DataFrame(
            {
                "episode_id": df[episode_col],
                "row_number": np.arange(len(df), dtype=np.int64),
                "value": pd.to_numeric(df[col], errors="coerce"),
                "column_name": col,
            }
        )
        work["previous_value"] = work.groupby("episode_id", sort=False)["value"].shift(1)
        work["delta_value"] = work["value"] - work["previous_value"]

        if date_col is not None and date_col in df.columns:
            work["date"] = df[date_col]
            work["previous_date"] = work.groupby("episode_id", sort=False)["date"].shift(1)
            work["delta_date_days"] = (work["date"] - work["previous_date"]).dt.days
        else:
            work["date"] = pd.NaT
            work["previous_date"] = pd.NaT
            work["delta_date_days"] = np.nan

        # Rule 1: non-decreasing holding period.
        non_decreasing_issue = (
            work["delta_value"].notna() & (work["delta_value"] < 0)
        )
        if non_decreasing_issue.any():
            part = work.loc[
                non_decreasing_issue,
                [
                    "episode_id",
                    "row_number",
                    "column_name",
                    "date",
                    "previous_date",
                    "value",
                    "previous_value",
                    "delta_value",
                    "delta_date_days",
                ],
            ].copy()
            part["issue_type"] = "holding_not_non_decreasing"
            issues_all.append(part)

        # Rule 2: holding period increments should match date increments where possible.
        if date_col is not None and date_col in df.columns:
            consistency_issue = (
                work["delta_value"].notna()
                & work["delta_date_days"].notna()
                & (work["delta_date_days"] >= 0)
                & (np.abs(work["delta_value"] - work["delta_date_days"]) > 1e-9)
            )
            if consistency_issue.any():
                part = work.loc[
                    consistency_issue,
                    [
                        "episode_id",
                        "row_number",
                        "column_name",
                        "date",
                        "previous_date",
                        "value",
                        "previous_value",
                        "delta_value",
                        "delta_date_days",
                    ],
                ].copy()
                part["issue_type"] = "holding_delta_vs_date_delta_mismatch"
                issues_all.append(part)

    if issues_all:
        issues = pd.concat(issues_all, ignore_index=True)
        issues = issues.sort_values(["episode_id", "row_number", "column_name"], kind="stable")
    else:
        issues = pd.DataFrame(
            columns=[
                "episode_id",
                "row_number",
                "column_name",
                "date",
                "previous_date",
                "value",
                "previous_value",
                "delta_value",
                "delta_date_days",
                "issue_type",
            ]
        )

    write_csv(issues, out_path)
    issue_rows = int(len(issues))
    episodes_with_issues = int(issues["episode_id"].nunique()) if issue_rows > 0 else 0
    status = "pass" if issue_rows == 0 else "fail"
    message = (
        f"Holding-period checks passed for columns: {holding_cols}"
        if status == "pass"
        else f"Found {issue_rows} holding-period issue rows across {episodes_with_issues} episodes."
    )
    return {
        "name": "holding_period_monotonicity",
        "status": status,
        "message": message,
        "output_file": out_path.name,
        "issue_rows": issue_rows,
        "episodes_with_issues": episodes_with_issues,
        "columns_checked": holding_cols,
    }


def detect_trigger_date_column(df: pd.DataFrame) -> str | None:
    if "trigger_date" in df.columns:
        return "trigger_date"
    candidates = [
        col
        for col in df.columns
        if "trigger" in col.lower() and "date" in col.lower()
    ]
    return candidates[0] if candidates else None


def detect_tax_transition_column(df: pd.DataFrame) -> str | None:
    if "tax_transition_date" in df.columns:
        return "tax_transition_date"
    candidates = [
        col
        for col in df.columns
        if "transition" in col.lower() and "date" in col.lower()
    ]
    return candidates[0] if candidates else None


def check_critical_missingness(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    out_path = output_dir / CRITICAL_MISSINGNESS_FILE

    expected_core = [
        "episode_id",
        date_col,
        "trigger_date",
        "tax_transition_date",
        "holding_period_days",
        "unrealized_gains_pct",
    ]
    expected_core = [x for x in expected_core if x is not None]

    purchase_like = [
        col
        for col in df.columns
        if re.search(r"(purchase.*price|entry.*price|cost.*basis|buy.*price)", col.lower())
    ]
    close_like = [
        col
        for col in df.columns
        if re.search(r"(^adj_close$|(^|_)adj_close$|(^|_)close$)", col.lower())
    ]

    critical_cols: list[str] = []
    for col in expected_core + purchase_like + close_like:
        if col in df.columns and col not in critical_cols:
            critical_cols.append(col)

    rows: list[dict[str, Any]] = []
    for col in critical_cols:
        null_count = int(df[col].isna().sum())
        null_pct = float((null_count / len(df) * 100.0) if len(df) else 0.0)

        if null_count == 0:
            severity = "no issue"
            note = "No missing values."
        elif col in {"episode_id", date_col}:
            severity = "critical issue"
            note = "Primary key/time column contains missing values."
        elif null_pct >= 5.0:
            severity = "critical issue"
            note = "High missingness in critical field."
        else:
            severity = "minor issue"
            note = "Low-to-moderate missingness in critical field."

        rows.append(
            {
                "column_name": col,
                "null_count": null_count,
                "null_pct": round(null_pct, 6),
                "severity": severity,
                "note": note,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["severity", "null_pct", "column_name"], ascending=[True, False, True])
    write_csv(summary, out_path)

    unavailable = [
        col
        for col in ["episode_id", date_col, "trigger_date", "tax_transition_date", "holding_period_days", "unrealized_gains_pct"]
        if col is not None and col not in df.columns
    ]

    critical_issues = int((summary["severity"] == "critical issue").sum()) if not summary.empty else 0
    minor_issues = int((summary["severity"] == "minor issue").sum()) if not summary.empty else 0

    if critical_issues > 0:
        status = "fail"
    elif minor_issues > 0:
        status = "needs_manual_review"
    else:
        status = "pass"

    message = (
        f"Critical missingness summary generated for {len(critical_cols)} columns; "
        f"critical={critical_issues}, minor={minor_issues}."
    )
    if unavailable:
        message += f" Skipped unavailable expected columns: {unavailable}."

    return {
        "name": "critical_missingness",
        "status": status,
        "message": message,
        "output_file": out_path.name,
        "critical_columns_checked": critical_cols,
        "unavailable_expected_columns": unavailable,
        "critical_issue_count": critical_issues,
        "minor_issue_count": minor_issues,
    }


def check_trigger_date_consistency(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    out_path = output_dir / TRIGGER_DATE_ISSUES_FILE
    trigger_col = detect_trigger_date_column(df)

    empty_cols = [
        "episode_id",
        "row_number",
        "date",
        "trigger_date",
        "issue_type",
        "detail",
    ]
    if episode_col is None or date_col is None or trigger_col is None:
        write_csv(pd.DataFrame(columns=empty_cols), out_path)
        return {
            "name": "trigger_date_consistency",
            "status": "skipped",
            "message": "episode_id/date/trigger_date columns not all available.",
            "output_file": out_path.name,
            "issue_rows": 0,
            "episodes_with_issues": 0,
            "trigger_column": trigger_col,
        }

    work = pd.DataFrame(
        {
            "episode_id": df[episode_col],
            "date": df[date_col],
            "trigger_date": df[trigger_col],
            "row_number": np.arange(len(df), dtype=np.int64),
        }
    )

    issues: list[pd.DataFrame] = []

    # Trigger date should be constant within episode.
    nunique_trigger = work.groupby("episode_id", sort=False)["trigger_date"].nunique(dropna=False)
    not_constant_eps = nunique_trigger[nunique_trigger > 1].index
    if len(not_constant_eps) > 0:
        part = pd.DataFrame({"episode_id": list(not_constant_eps)})
        part["row_number"] = np.nan
        part["date"] = pd.NaT
        part["trigger_date"] = pd.NaT
        part["issue_type"] = "trigger_not_constant_within_episode"
        part["detail"] = "More than one trigger_date value found in the same episode."
        issues.append(part)

    # Trigger date should not be after row date.
    mask_trigger_after_row = (
        work["trigger_date"].notna()
        & work["date"].notna()
        & (work["trigger_date"] > work["date"])
    )
    if mask_trigger_after_row.any():
        part = work.loc[
            mask_trigger_after_row,
            ["episode_id", "row_number", "date", "trigger_date"],
        ].copy()
        part["issue_type"] = "trigger_after_row_date"
        part["detail"] = "trigger_date occurs after the per-step date."
        issues.append(part)

    # First row date relation to trigger_date.
    first_rows = (
        work.groupby("episode_id", sort=False)
        .first()
        .reset_index()
    )
    first_rows = first_rows[first_rows["date"].notna() & first_rows["trigger_date"].notna()].copy()

    before_mask = first_rows["date"] < first_rows["trigger_date"]
    if before_mask.any():
        part = first_rows.loc[before_mask, ["episode_id", "row_number", "date", "trigger_date"]].copy()
        part["issue_type"] = "first_row_before_trigger"
        part["detail"] = "First observed row date is before trigger_date."
        issues.append(part)

    after_mask = first_rows["date"] > first_rows["trigger_date"]
    if after_mask.any():
        part = first_rows.loc[after_mask, ["episode_id", "row_number", "date", "trigger_date"]].copy()
        part["issue_type"] = "first_row_after_trigger_manual_review"
        part["detail"] = (
            "First observed row date is after trigger_date; this can be valid if early rows were dropped."
        )
        issues.append(part)

    if issues:
        issues_df = pd.concat(issues, ignore_index=True)
        issues_df = issues_df.sort_values(["episode_id", "issue_type", "row_number"], kind="stable")
    else:
        issues_df = pd.DataFrame(columns=empty_cols)

    write_csv(issues_df, out_path)

    issue_rows = int(len(issues_df))
    episodes_with_issues = int(issues_df["episode_id"].nunique()) if issue_rows > 0 else 0
    hard_issue_types = {
        "trigger_not_constant_within_episode",
        "trigger_after_row_date",
        "first_row_before_trigger",
    }
    has_hard_issues = bool(
        issue_rows > 0 and issues_df["issue_type"].isin(hard_issue_types).any()
    )
    has_manual_only = bool(
        issue_rows > 0
        and not has_hard_issues
        and issues_df["issue_type"].eq("first_row_after_trigger_manual_review").all()
    )

    if has_hard_issues:
        status = "fail"
    elif has_manual_only:
        status = "needs_manual_review"
    else:
        status = "pass"

    message = (
        "Trigger date consistency checks passed."
        if status == "pass"
        else f"Found {issue_rows} trigger-date issue rows across {episodes_with_issues} episodes."
    )

    return {
        "name": "trigger_date_consistency",
        "status": status,
        "message": message,
        "output_file": out_path.name,
        "issue_rows": issue_rows,
        "episodes_with_issues": episodes_with_issues,
        "trigger_column": trigger_col,
    }


def check_tax_transition_consistency(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    out_path = output_dir / TAX_TRANSITION_ISSUES_FILE
    tax_col = detect_tax_transition_column(df)
    trigger_col = detect_trigger_date_column(df)
    holding_col = "holding_period_days" if "holding_period_days" in df.columns else None

    empty_cols = [
        "episode_id",
        "row_number",
        "date",
        "tax_transition_date",
        "trigger_date",
        "holding_period_days",
        "days_to_transition",
        "issue_type",
        "detail",
    ]

    if episode_col is None or date_col is None or tax_col is None:
        write_csv(pd.DataFrame(columns=empty_cols), out_path)
        return {
            "name": "tax_transition_consistency",
            "status": "skipped",
            "message": "episode_id/date/tax_transition_date columns not all available.",
            "output_file": out_path.name,
            "issue_rows": 0,
            "episodes_with_issues": 0,
            "tax_transition_column": tax_col,
        }

    work = pd.DataFrame(
        {
            "episode_id": df[episode_col],
            "date": df[date_col],
            "tax_transition_date": df[tax_col],
            "row_number": np.arange(len(df), dtype=np.int64),
        }
    )
    if trigger_col is not None:
        work["trigger_date"] = df[trigger_col]
    else:
        work["trigger_date"] = pd.NaT

    if holding_col is not None:
        work["holding_period_days"] = pd.to_numeric(df[holding_col], errors="coerce")
    else:
        work["holding_period_days"] = np.nan

    work["days_to_transition"] = (work["tax_transition_date"] - work["date"]).dt.days

    issues: list[pd.DataFrame] = []

    # Constant tax_transition_date in episode.
    nunique_tax = work.groupby("episode_id", sort=False)["tax_transition_date"].nunique(dropna=False)
    eps_not_constant = nunique_tax[nunique_tax > 1].index
    if len(eps_not_constant) > 0:
        part = pd.DataFrame({"episode_id": list(eps_not_constant)})
        part["row_number"] = np.nan
        part["date"] = pd.NaT
        part["tax_transition_date"] = pd.NaT
        part["trigger_date"] = pd.NaT
        part["holding_period_days"] = np.nan
        part["days_to_transition"] = np.nan
        part["issue_type"] = "tax_transition_not_constant_within_episode"
        part["detail"] = "More than one tax_transition_date value found in the same episode."
        issues.append(part)

    # tax_transition_date should be on or after trigger_date when both exist.
    mask_tax_before_trigger = (
        work["tax_transition_date"].notna()
        & work["trigger_date"].notna()
        & (work["tax_transition_date"] < work["trigger_date"])
    )
    if mask_tax_before_trigger.any():
        part = work.loc[
            mask_tax_before_trigger,
            [
                "episode_id",
                "row_number",
                "date",
                "tax_transition_date",
                "trigger_date",
                "holding_period_days",
                "days_to_transition",
            ],
        ].copy()
        part["issue_type"] = "tax_transition_before_trigger"
        part["detail"] = "tax_transition_date is earlier than trigger_date."
        issues.append(part)

    # days_to_transition should decline with date increments.
    work["prev_days_to_transition"] = work.groupby("episode_id", sort=False)["days_to_transition"].shift(1)
    work["prev_date"] = work.groupby("episode_id", sort=False)["date"].shift(1)
    work["delta_days_to_transition"] = work["days_to_transition"] - work["prev_days_to_transition"]
    work["delta_date_days"] = (work["date"] - work["prev_date"]).dt.days

    mismatch_mask = (
        work["delta_days_to_transition"].notna()
        & work["delta_date_days"].notna()
        & (work["delta_date_days"] >= 0)
        & (np.abs(work["delta_days_to_transition"] + work["delta_date_days"]) > 1e-9)
    )
    if mismatch_mask.any():
        part = work.loc[
            mismatch_mask,
            [
                "episode_id",
                "row_number",
                "date",
                "tax_transition_date",
                "trigger_date",
                "holding_period_days",
                "days_to_transition",
            ],
        ].copy()
        part["issue_type"] = "days_to_transition_delta_mismatch"
        part["detail"] = "days_to_transition change does not match date delta."
        issues.append(part)

    # If holding_period_days exists, check holding + days_to_transition constancy within episode.
    if holding_col is not None:
        work["holding_plus_days_to_transition"] = (
            work["holding_period_days"] + work["days_to_transition"]
        )
        baseline = work.groupby("episode_id", sort=False)["holding_plus_days_to_transition"].transform(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
        )
        constancy_mask = (
            work["holding_plus_days_to_transition"].notna()
            & baseline.notna()
            & (np.abs(work["holding_plus_days_to_transition"] - baseline) > 1e-9)
        )
        if constancy_mask.any():
            part = work.loc[
                constancy_mask,
                [
                    "episode_id",
                    "row_number",
                    "date",
                    "tax_transition_date",
                    "trigger_date",
                    "holding_period_days",
                    "days_to_transition",
                ],
            ].copy()
            part["issue_type"] = "holding_plus_transition_not_constant"
            part["detail"] = (
                "holding_period_days + days_to_transition varies within episode."
            )
            issues.append(part)

    if issues:
        issues_df = pd.concat(issues, ignore_index=True)
        issues_df = issues_df.sort_values(["episode_id", "issue_type", "row_number"], kind="stable")
    else:
        issues_df = pd.DataFrame(columns=empty_cols)

    write_csv(issues_df, out_path)
    issue_rows = int(len(issues_df))
    episodes_with_issues = int(issues_df["episode_id"].nunique()) if issue_rows > 0 else 0
    status = "pass" if issue_rows == 0 else "fail"
    message = (
        "Tax transition consistency checks passed."
        if status == "pass"
        else f"Found {issue_rows} tax-transition issue rows across {episodes_with_issues} episodes."
    )
    return {
        "name": "tax_transition_consistency",
        "status": status,
        "message": message,
        "output_file": out_path.name,
        "issue_rows": issue_rows,
        "episodes_with_issues": episodes_with_issues,
        "tax_transition_column": tax_col,
    }


def detect_purchase_price_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"(purchase.*price|entry.*price|cost.*basis|buy.*price)")
    cols = [col for col in df.columns if pattern.search(col.lower())]
    if "simulated_purchase_price" in df.columns and "simulated_purchase_price" not in cols:
        cols.insert(0, "simulated_purchase_price")
    return cols


def detect_current_price_columns(df: pd.DataFrame, purchase_cols: list[str]) -> list[str]:
    candidates: list[str] = []
    priority = ["adj_close", "close"]

    for col in priority:
        if col in df.columns and col not in purchase_cols:
            candidates.append(col)

    for col in df.columns:
        lower = col.lower()
        if col in purchase_cols:
            continue
        if lower in {"unrealized_gains_pct"}:
            continue
        if lower.endswith("_close") and col not in candidates:
            candidates.append(col)
    return candidates


def check_unrealized_gain_consistency(
    df: pd.DataFrame,
    episode_col: str | None,
    date_col: str | None,
    output_dir: Path,
    diagnostic_sample_size: int,
) -> dict[str, Any]:
    diag_path = output_dir / UNREALIZED_GAIN_DIAGNOSTICS_FILE
    score_path = output_dir / UNREALIZED_GAIN_SCORES_FILE

    target_col = "unrealized_gains_pct" if "unrealized_gains_pct" in df.columns else None
    if target_col is None:
        write_csv(
            pd.DataFrame(
                columns=[
                    "episode_id",
                    "date",
                    "target_unrealized_gains_pct",
                    "candidate_purchase_col",
                    "candidate_current_col",
                    "formula_name",
                    "calculated_value",
                    "abs_error",
                ]
            ),
            diag_path,
        )
        write_csv(
            pd.DataFrame(
                columns=[
                    "purchase_col",
                    "current_col",
                    "formula_name",
                    "valid_rows",
                    "mae",
                    "rmse",
                    "corr",
                    "exact_match_ratio_1e_9",
                    "near_match_ratio_1e_6",
                ]
            ),
            score_path,
        )
        return {
            "name": "unrealized_gain_consistency",
            "status": "skipped",
            "message": "unrealized_gains_pct column not available.",
            "output_file": diag_path.name,
            "scores_file": score_path.name,
        }

    purchase_cols = detect_purchase_price_columns(df)
    current_cols = detect_current_price_columns(df, purchase_cols)

    if not purchase_cols or not current_cols:
        write_csv(
            pd.DataFrame(
                columns=[
                    "episode_id",
                    "date",
                    "target_unrealized_gains_pct",
                    "candidate_purchase_col",
                    "candidate_current_col",
                    "formula_name",
                    "calculated_value",
                    "abs_error",
                ]
            ),
            diag_path,
        )
        write_csv(
            pd.DataFrame(
                columns=[
                    "purchase_col",
                    "current_col",
                    "formula_name",
                    "valid_rows",
                    "mae",
                    "rmse",
                    "corr",
                    "exact_match_ratio_1e_9",
                    "near_match_ratio_1e_6",
                ]
            ),
            score_path,
        )
        return {
            "name": "unrealized_gain_consistency",
            "status": "needs_manual_review",
            "message": (
                "Could not infer both purchase-price and current-price candidates. "
                f"purchase_cols={purchase_cols}, current_cols={current_cols}"
            ),
            "output_file": diag_path.name,
            "scores_file": score_path.name,
        }

    target = pd.to_numeric(df[target_col], errors="coerce")
    metric_rows: list[dict[str, Any]] = []

    for purchase_col in purchase_cols:
        purchase = pd.to_numeric(df[purchase_col], errors="coerce")
        for current_col in current_cols:
            current = pd.to_numeric(df[current_col], errors="coerce")
            valid = target.notna() & purchase.notna() & current.notna() & (purchase != 0)
            valid_rows = int(valid.sum())
            if valid_rows == 0:
                continue

            y = target.loc[valid].astype(float)
            p = purchase.loc[valid].astype(float)
            c = current.loc[valid].astype(float)
            calc_fraction = (c - p) / p
            calc_percent = calc_fraction * 100.0

            formulas = {
                "fraction_(current-purchase)/purchase": calc_fraction,
                "percent_((current/purchase)-1)*100": calc_percent,
            }

            for formula_name, calc in formulas.items():
                err = y - calc
                abs_err = np.abs(err)
                mae = float(abs_err.mean())
                rmse = float(np.sqrt(np.mean(np.square(err))))
                corr = float(y.corr(calc)) if valid_rows > 1 else np.nan
                exact_match_ratio = float((abs_err <= 1e-9).mean())
                near_match_ratio = float((abs_err <= 1e-6).mean())
                metric_rows.append(
                    {
                        "purchase_col": purchase_col,
                        "current_col": current_col,
                        "formula_name": formula_name,
                        "valid_rows": valid_rows,
                        "mae": mae,
                        "rmse": rmse,
                        "corr": corr,
                        "exact_match_ratio_1e_9": exact_match_ratio,
                        "near_match_ratio_1e_6": near_match_ratio,
                    }
                )

    scores = pd.DataFrame(metric_rows)
    if not scores.empty:
        scores = scores.sort_values(
            ["mae", "rmse", "purchase_col", "current_col", "formula_name"],
            kind="stable",
        ).reset_index(drop=True)
    write_csv(scores, score_path)

    if scores.empty:
        write_csv(
            pd.DataFrame(
                columns=[
                    "episode_id",
                    "date",
                    "target_unrealized_gains_pct",
                    "candidate_purchase_col",
                    "candidate_current_col",
                    "formula_name",
                    "calculated_value",
                    "abs_error",
                ]
            ),
            diag_path,
        )
        return {
            "name": "unrealized_gain_consistency",
            "status": "needs_manual_review",
            "message": "No valid rows to compare unrealized gain formulas.",
            "output_file": diag_path.name,
            "scores_file": score_path.name,
        }

    best = scores.iloc[0]
    best_purchase_col = str(best["purchase_col"])
    best_current_col = str(best["current_col"])
    best_formula = str(best["formula_name"])

    purchase_best = pd.to_numeric(df[best_purchase_col], errors="coerce")
    current_best = pd.to_numeric(df[best_current_col], errors="coerce")
    valid_best = (
        target.notna() & purchase_best.notna() & current_best.notna() & (purchase_best != 0)
    )
    if best_formula == "fraction_(current-purchase)/purchase":
        calc_best = (current_best - purchase_best) / purchase_best
    else:
        calc_best = ((current_best / purchase_best) - 1.0) * 100.0

    diag = pd.DataFrame(
        {
            "target_unrealized_gains_pct": target,
            "candidate_purchase_col": best_purchase_col,
            "candidate_current_col": best_current_col,
            "formula_name": best_formula,
            "calculated_value": calc_best,
            "abs_error": np.abs(target - calc_best),
        }
    )
    if episode_col is not None and episode_col in df.columns:
        diag.insert(0, "episode_id", df[episode_col])
    else:
        diag.insert(0, "episode_id", "")

    if date_col is not None and date_col in df.columns:
        diag.insert(1, "date", df[date_col])
    else:
        diag.insert(1, "date", pd.NaT)

    diag = diag.loc[valid_best].copy()
    diag = diag.head(diagnostic_sample_size)
    write_csv(diag, diag_path)

    best_mae = float(best["mae"])
    if best_mae <= 1e-9:
        status = "pass"
    elif best_mae <= 1e-4:
        status = "pass"
    elif best_mae <= 1e-2:
        status = "needs_manual_review"
    else:
        status = "fail"

    if best_formula == "fraction_(current-purchase)/purchase":
        scale_note = (
            "This indicates the stored values are fractional returns "
            "(for display as percentage terms, multiply by 100)."
        )
    else:
        scale_note = (
            "This indicates the stored values are already in percentage points."
        )

    message = (
        "Best unrealized-gain interpretation: "
        f"{best_formula} using purchase='{best_purchase_col}' and current='{best_current_col}'. "
        f"MAE={best_mae:.6g}, valid_rows={int(best['valid_rows'])}. "
        f"{scale_note}"
    )
    return {
        "name": "unrealized_gain_consistency",
        "status": status,
        "message": message,
        "output_file": diag_path.name,
        "scores_file": score_path.name,
        "best_formula": best_formula,
        "best_purchase_col": best_purchase_col,
        "best_current_col": best_current_col,
        "best_mae": best_mae,
        "scale_note": scale_note,
    }


def render_markdown_report(
    *,
    input_path: Path,
    df: pd.DataFrame,
    date_cols: list[str],
    main_date_col: str | None,
    main_date_explanation: str,
    strict_numeric_cols: list[str],
    numeric_like_cols: list[str],
    classification_df: pd.DataFrame,
    check_results: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    out_path = output_dir / VALIDATION_REPORT_FILE

    rows = len(df)
    cols = len(df.columns)
    class_counts = classification_df["usage_class"].value_counts().to_dict()

    lines: list[str] = []
    lines.append("# Parquet Validation Report")
    lines.append("")
    lines.append("## 1. Dataset overview")
    lines.append("")
    lines.append(f"- Input file: `{input_path}`")
    lines.append(f"- Row count: **{rows:,}**")
    lines.append(f"- Column count: **{cols:,}**")
    lines.append(f"- Date-like columns inferred: {date_cols if date_cols else 'None'}")
    lines.append(f"- Main per-step date column: `{main_date_col}`" if main_date_col else "- Main per-step date column: Not detected")
    lines.append(f"- Main date selection rationale: {main_date_explanation}")
    lines.append(f"- Strict numeric columns: {len(strict_numeric_cols)}")
    lines.append(f"- Numeric-like non-numeric columns (heuristic): {numeric_like_cols if numeric_like_cols else 'None'}")
    lines.append("")
    lines.append("## 2. Column classification summary")
    lines.append("")
    for usage_class in ["observation_feature", "tax_variable", "identifier", "simulator_only", "excluded"]:
        lines.append(f"- `{usage_class}`: {class_counts.get(usage_class, 0)} columns")
    lines.append(f"- Full classification CSV: `{COLUMN_CLASSIFICATION_FILE}`")
    lines.append(f"- Full data dictionary CSV: `{DATA_DICTIONARY_FILE}`")
    lines.append("")
    lines.append("## 3. Key validation checks")
    lines.append("")
    lines.append("| Check | Status | Notes | Diagnostics |")
    lines.append("|---|---|---|---|")
    for result in check_results:
        diagnostics = result.get("output_file", "-")
        if result.get("scores_file"):
            diagnostics = f"{diagnostics}, {result['scores_file']}"
        lines.append(
            f"| {result.get('name')} | {result.get('status')} | {result.get('message')} | `{diagnostics}` |"
        )
    lines.append("")
    lines.append("## 4. Issues found")
    lines.append("")

    issue_results = [
        r
        for r in check_results
        if r.get("status") in {"fail", "needs_manual_review"}
    ]
    if not issue_results:
        lines.append("- No blocking issues detected by automated checks.")
    else:
        for result in issue_results:
            lines.append(f"- `{result.get('name')}` -> **{result.get('status')}**: {result.get('message')}")
    lines.append("")
    lines.append("## 5. Candidate fixes or manual review items")
    lines.append("")
    lines.append("- Confirm semantic intent for columns classified as `excluded` before training.")
    lines.append("- Confirm that dashboard/notebook display logic matches the detected unrealized-gain scale (fraction vs percentage points).")
    lines.append("- If `first_row_after_trigger` issues appear, verify whether rows before first valid feature vector were intentionally dropped.")
    lines.append("- If any missingness appears in critical fields, decide whether to impute, drop, or regenerate episodes upstream.")
    lines.append("")
    lines.append("## 6. Overall assessment")
    lines.append("")

    if any(r.get("status") == "fail" for r in check_results):
        overall = "Dataset has one or more failed validation checks; manual remediation is recommended before modeling."
    elif any(r.get("status") == "needs_manual_review" for r in check_results):
        overall = "No hard failures found, but some checks need manual confirmation."
    else:
        overall = "Automated checks passed for the available fields."
    lines.append(f"- {overall}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {input_path}. "
            "Pass a valid path via --input."
        )

    df = pd.read_parquet(input_path)

    # A) load and profile
    print("=== Dataset basic info ===")
    print(f"Input file: {input_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Column list: {list(df.columns)}")
    print("Dtypes:")
    print(df.dtypes.to_string())

    date_info = infer_date_columns(df, args.max_infer_sample)
    df, converted_date_cols, date_conversion_notes = convert_date_columns(df, date_info)
    strict_numeric_cols, numeric_like_cols = infer_numeric_columns(df, args.max_infer_sample)

    inferred_date_cols = [col for col, info in date_info.items() if info["is_likely_date"]]
    main_date_col, main_date_explanation = choose_main_date_column(df, inferred_date_cols)

    print("\n=== Inference summary ===")
    print(f"Inferred date-like columns: {inferred_date_cols}")
    print(f"Converted date columns: {converted_date_cols}")
    print(f"Date conversion notes: {date_conversion_notes}")
    print(f"Main date column: {main_date_col}")
    print(f"Main date rationale: {main_date_explanation}")
    print(f"Strict numeric columns ({len(strict_numeric_cols)}): {strict_numeric_cols}")
    print(f"Numeric-like columns ({len(numeric_like_cols)}): {numeric_like_cols}")

    # B) data dictionary
    data_dictionary_df = build_data_dictionary(
        df=df,
        date_cols=inferred_date_cols,
        max_example_values=args.max_example_values,
    )
    data_dictionary_path = output_dir / DATA_DICTIONARY_FILE
    write_csv(data_dictionary_df, data_dictionary_path)

    # C) column classification
    classification_df = build_column_classification(
        df=df,
        date_cols=inferred_date_cols,
        main_date_col=main_date_col,
    )
    classification_path = output_dir / COLUMN_CLASSIFICATION_FILE
    write_csv(classification_df, classification_path)

    # D) validation checks
    episode_col = "episode_id" if "episode_id" in df.columns else None

    check_results = []
    check_results.append(check_duplicate_episode_date(df, episode_col, main_date_col, output_dir))
    check_results.append(check_episode_date_order(df, episode_col, main_date_col, output_dir))
    check_results.append(check_holding_period(df, episode_col, main_date_col, output_dir))
    check_results.append(check_critical_missingness(df, episode_col, main_date_col, output_dir))
    check_results.append(check_trigger_date_consistency(df, episode_col, main_date_col, output_dir))
    check_results.append(check_tax_transition_consistency(df, episode_col, main_date_col, output_dir))
    check_results.append(
        check_unrealized_gain_consistency(
            df=df,
            episode_col=episode_col,
            date_col=main_date_col,
            output_dir=output_dir,
            diagnostic_sample_size=args.diagnostic_sample_size,
        )
    )

    # E) markdown report
    report_path = render_markdown_report(
        input_path=input_path,
        df=df,
        date_cols=inferred_date_cols,
        main_date_col=main_date_col,
        main_date_explanation=main_date_explanation,
        strict_numeric_cols=strict_numeric_cols,
        numeric_like_cols=numeric_like_cols,
        classification_df=classification_df,
        check_results=check_results,
        output_dir=output_dir,
    )

    produced_files = [
        data_dictionary_path.name,
        classification_path.name,
        report_path.name,
        DUPLICATE_EPISODE_DATE_FILE,
        EPISODE_ORDER_ISSUES_FILE,
        HOLDING_PERIOD_ISSUES_FILE,
        CRITICAL_MISSINGNESS_FILE,
        TRIGGER_DATE_ISSUES_FILE,
        TAX_TRANSITION_ISSUES_FILE,
        UNREALIZED_GAIN_DIAGNOSTICS_FILE,
        UNREALIZED_GAIN_SCORES_FILE,
    ]

    print("\n=== Output files ===")
    for name in produced_files:
        print(f"- {output_dir / name}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
