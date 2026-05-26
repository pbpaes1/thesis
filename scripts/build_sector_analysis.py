"""Build sector-level diagnostic analysis for the frozen final run.

Run from the project root:
    python scripts/build_sector_analysis.py --config configs/train_reward_c_lite_v5.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to run sector analysis.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_sector_metadata import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_METADATA_PATH,
    build_metadata,
    extract_tickers_from_artifacts,
    relative_path,
    write_notes as write_metadata_notes,
)


FINAL_RUN_NAME = "train_reward_c_lite_v5_full"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_reward_c_lite_v5.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / FINAL_RUN_NAME
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "quant_analysis" / "sector_analysis"
PREFERRED_POLICY = "trained_dqn_first_sale_margin_0p070_normal_0p020"
POLICIES = [
    PREFERRED_POLICY,
    "hold_to_terminal",
    "sell_immediately",
    "sell_half_then_hold",
]
POLICY_LABELS = {
    PREFERRED_POLICY: "preferred",
    "hold_to_terminal": "hold_to_terminal",
    "sell_immediately": "sell_immediately",
    "sell_half_then_hold": "sell_half_then_hold",
}
PRIMARY_SPLITS = ["validation", "test"]
LEGACY_SHARPE_COLUMNS = {
    "sharpe_episode",
    "sortino_episode",
    "episode_step_sharpe",
    "pooled_step_sharpe",
    "full_horizon_annualized_sharpe",
    "invested_period_annualized_sharpe",
    "mean_TA_EAAT_Sharpe",
    "std_TA_EAAT_Sharpe",
    "reward_path_sharpe",
    "step_return_proxy",
}
REQUIRED_OUTPUT_COLUMNS = [
    "split",
    "sector",
    "num_episodes",
    "small_group_warning",
    "preferred_mean_final_after_tax_value",
    "hold_to_terminal_mean_final_after_tax_value",
    "sell_immediately_mean_final_after_tax_value",
    "sell_half_then_hold_mean_final_after_tax_value",
    "preferred_minus_hold_mean",
    "preferred_minus_sell_immediately_mean",
    "preferred_minus_sell_half_mean",
    "preferred_win_rate_vs_hold",
    "preferred_win_rate_vs_sell_immediately",
    "preferred_win_rate_vs_sell_half",
    "preferred_mean_tax_paid",
    "preferred_median_tax_paid",
    "preferred_mean_effective_tax_rate",
    "preferred_mean_short_term_sold_fraction",
    "preferred_mean_long_term_sold_fraction",
    "preferred_no_cut_pct",
    "preferred_discretionary_sale_pct",
    "preferred_average_days_to_first_sale",
    "preferred_median_days_to_first_sale",
    "preferred_median_EAAT_Sharpe",
    "hold_to_terminal_median_EAAT_Sharpe",
    "sell_immediately_median_EAAT_Sharpe",
    "sell_half_then_hold_median_EAAT_Sharpe",
    "preferred_median_TA_EAAT_Sharpe",
    "hold_to_terminal_median_TA_EAAT_Sharpe",
    "sell_immediately_median_TA_EAAT_Sharpe",
    "sell_half_then_hold_median_TA_EAAT_Sharpe",
    "preferred_minus_hold_median_EAAT_Sharpe",
    "preferred_EAAT_Sharpe_win_rate_vs_hold",
    "preferred_minus_hold_median_TA_EAAT_Sharpe",
    "preferred_TA_EAAT_Sharpe_win_rate_vs_hold",
    "preferred_minus_sell_immediately_median_EAAT_Sharpe",
    "preferred_minus_sell_half_median_EAAT_Sharpe",
    "preferred_minus_sell_immediately_median_TA_EAAT_Sharpe",
    "preferred_minus_sell_half_median_TA_EAAT_Sharpe",
]


class Manifest:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, path: Path, file_type: str, description: str) -> None:
        self.rows.append(
            {
                "path": relative_path(path),
                "type": file_type,
                "description": description,
            }
        )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {relative_path(path)}.")
    return data


def nested_get(mapping: dict[str, Any], dotted_path: str) -> Any:
    value: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing config key: {dotted_path}")
        value = value[part]
    return value


def resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def verify_run(config_path: Path, output_dir: Path) -> tuple[dict[str, Any], Path]:
    config = load_yaml(config_path)
    run_dir = resolve_project_path(nested_get(config, "logging.output_dir"))
    if run_dir.name != FINAL_RUN_NAME:
        raise ValueError(
            f"Expected frozen run {FINAL_RUN_NAME!r}, got {relative_path(run_dir)!r}."
        )
    if output_dir.resolve() != (run_dir / "quant_analysis" / "sector_analysis").resolve():
        raise ValueError(
            "Sector analysis output must be under the frozen run quant_analysis "
            f"folder: {relative_path(run_dir / 'quant_analysis' / 'sector_analysis')}."
        )
    return config, run_dir


def markdown_table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6g}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    lines = [
        "| "
        + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
            + " |"
        )
    return "\n".join(lines) + "\n"


def save_table(
    df: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
    manifest: Manifest,
    description: str,
) -> None:
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table_text(df), encoding="utf-8")
    manifest.add(csv_path, "csv", description)
    manifest.add(md_path, "md", description)


def save_plot(path: Path, manifest: Manifest, description: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    manifest.add(path, "png", description)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else np.nan


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.median()) if values.notna().any() else np.nan


def safe_win_rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    return float(values.gt(0).mean()) if len(values) else np.nan


def extract_ticker_from_episode_data(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for column in ["ticker", "symbol", "asset", "asset_symbol", "stock", "stock_ticker"]:
        if column in df.columns:
            return df[column].astype(str).str.strip(), f"direct column {column}"
    if "episode_id" not in df.columns:
        raise ValueError("Cannot extract ticker: no ticker-like column or episode_id.")
    return (
        df["episode_id"].astype(str).str.split("_", n=1).str[0].str.strip(),
        "episode_id prefix before first underscore",
    )


def load_episode_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "baselines" / "baseline_episode_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline episode metrics: {relative_path(path)}")
    df = pd.read_csv(path, low_memory=False)
    required = [
        "split",
        "policy_name",
        "episode_id",
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "episode_cut_occurred",
        "num_discretionary_sales",
        "days_to_first_sale",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Sector analysis missing required episode metric column(s): "
            + ", ".join(missing)
        )
    numeric_columns = [
        "episode_final_after_tax_total_value",
        "total_tax_paid",
        "mean_effective_tax_rate_on_sales",
        "pct_episode_position_sold_short_term",
        "pct_episode_position_sold_long_term",
        "num_discretionary_sales",
        "days_to_first_sale",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["episode_cut_occurred"] = df["episode_cut_occurred"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if pd.notna(value)
        else False
    )
    df["ticker"], _ = extract_ticker_from_episode_data(df)
    return df


def missing_split_policy_rows(df: pd.DataFrame, splits: list[str]) -> list[str]:
    missing: list[str] = []
    for split in splits:
        for policy in POLICIES:
            if df[df["split"].eq(split) & df["policy_name"].eq(policy)].empty:
                missing.append(f"{split}/{policy}")
    return missing


def validate_policy_rows(df: pd.DataFrame, splits: list[str]) -> None:
    missing = missing_split_policy_rows(df, splits)
    if missing:
        raise ValueError(
            "Sector analysis missing preferred or benchmark policy rows: "
            + ", ".join(missing)
        )
    duplicate_count = int(
        df[df["split"].isin(splits) & df["policy_name"].isin(POLICIES)]
        .duplicated(["split", "policy_name", "episode_id"])
        .sum()
    )
    if duplicate_count:
        raise ValueError(
            "Sector analysis requires unique split/policy/episode rows; found "
            f"{duplicate_count} duplicates."
        )


def ensure_metadata(
    *,
    run_dir: Path,
    metadata_path: Path,
    force_refresh: bool,
    max_workers: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    required_tickers, extraction_notes = extract_tickers_from_artifacts(run_dir)
    if force_refresh or not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata, summary, reused_cache, metadata_extraction_notes = build_metadata(
            run_dir=run_dir,
            output_path=metadata_path,
            force_refresh=force_refresh,
            max_workers=max_workers,
            pause_seconds=0.0,
        )
        metadata.to_csv(metadata_path, index=False)
        notes_path = metadata_path.with_name(metadata_path.stem + "_notes.txt")
        write_metadata_notes(
            notes_path,
            output_path=metadata_path,
            run_dir=run_dir,
            extraction_notes=metadata_extraction_notes,
            summary=summary,
            reused_cache=reused_cache,
            force_refresh=force_refresh,
        )
    else:
        metadata = pd.read_csv(metadata_path, low_memory=False)
        if "ticker" not in metadata.columns:
            raise ValueError(f"Metadata file has no ticker column: {relative_path(metadata_path)}")
        missing_tickers = sorted(set(required_tickers) - set(metadata["ticker"].astype(str)))
        if missing_tickers:
            metadata, summary, reused_cache, metadata_extraction_notes = build_metadata(
                run_dir=run_dir,
                output_path=metadata_path,
                force_refresh=False,
                max_workers=max_workers,
                pause_seconds=0.0,
            )
            metadata.to_csv(metadata_path, index=False)
            notes_path = metadata_path.with_name(metadata_path.stem + "_notes.txt")
            write_metadata_notes(
                notes_path,
                output_path=metadata_path,
                run_dir=run_dir,
                extraction_notes=metadata_extraction_notes,
                summary=summary,
                reused_cache=reused_cache,
                force_refresh=False,
            )
    metadata = pd.read_csv(metadata_path, low_memory=False)
    for column in ["sector", "industry"]:
        if column not in metadata.columns:
            raise ValueError(
                f"Metadata file missing required column {column}: {relative_path(metadata_path)}"
            )
        metadata[column] = metadata[column].fillna("Unknown").replace("", "Unknown")
    required = metadata[metadata["ticker"].astype(str).isin(required_tickers)].copy()
    summary = {
        "num_tickers_found": len(required_tickers),
        "num_with_sector": int(required["sector"].ne("Unknown").sum()),
        "num_missing_sector": int(required["sector"].eq("Unknown").sum()),
        "num_failed": int(required.get("metadata_fetch_status", pd.Series(dtype=str)).eq("failed").sum()),
    }
    return metadata, summary, extraction_notes


def merge_metadata(episode_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["ticker", "sector", "industry"]
    missing = [column for column in required_columns if column not in metadata.columns]
    if missing:
        raise ValueError("Ticker metadata missing column(s): " + ", ".join(missing))
    meta = metadata[required_columns].copy()
    meta["ticker"] = meta["ticker"].astype(str).str.strip()
    meta = meta.drop_duplicates("ticker", keep="last")
    merged = episode_df.merge(meta, on="ticker", how="left", validate="many_to_one")
    if merged["sector"].isna().any():
        missing_tickers = sorted(merged.loc[merged["sector"].isna(), "ticker"].unique())
        raise ValueError(
            "Ticker metadata could not be joined for ticker(s): "
            + ", ".join(missing_tickers[:20])
        )
    merged["sector"] = merged["sector"].fillna("Unknown").replace("", "Unknown")
    merged["industry"] = merged["industry"].fillna("Unknown").replace("", "Unknown")
    if merged["sector"].isna().all():
        raise ValueError("Sector column is entirely missing after metadata join.")
    return merged


def load_episode_sharpe(output_dir: Path, splits: list[str]) -> pd.DataFrame:
    path = output_dir.parent / "step3b_episode_eaat_sharpe_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Sector analysis requires corrected Step 3B episode-level EAAT/TA-EAAT "
            f"metrics at {relative_path(path)}."
        )
    df = pd.read_csv(path, low_memory=False)
    required = ["split", "policy_name", "episode_id", "EAAT_Sharpe", "TA_EAAT_Sharpe"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Corrected Step 3B episode Sharpe file missing column(s): "
            + ", ".join(missing)
        )
    duplicate_count = int(df.duplicated(["split", "policy_name", "episode_id"]).sum())
    if duplicate_count:
        raise ValueError(
            "Corrected Step 3B episode Sharpe file has duplicate split/policy/episode "
            f"rows: {duplicate_count}."
        )
    missing_rows = missing_split_policy_rows(
        df[df["split"].isin(splits) & df["policy_name"].isin(POLICIES)],
        splits,
    )
    if missing_rows:
        raise ValueError(
            "Corrected Step 3B episode Sharpe file missing required policy rows: "
            + ", ".join(missing_rows)
        )
    out = df[df["split"].isin(splits) & df["policy_name"].isin(POLICIES)].copy()
    for column in ["EAAT_Sharpe", "TA_EAAT_Sharpe"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
    return out[required]


def merge_sharpe(episode_df: pd.DataFrame, sharpe_df: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    work = episode_df[episode_df["split"].isin(splits) & episode_df["policy_name"].isin(POLICIES)].copy()
    merged = work.merge(
        sharpe_df,
        on=["split", "policy_name", "episode_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged["_merge"].ne("both")
    if missing.any():
        sample = merged.loc[missing, ["split", "policy_name", "episode_id"]].head(10)
        raise ValueError(
            "Corrected EAAT/TA-EAAT episode-level metrics could not be merged for "
            f"{int(missing.sum())} rows. Sample: {sample.to_dict(orient='records')}"
        )
    return merged.drop(columns=["_merge"])


def paired_base_for_split(merged: pd.DataFrame, split: str) -> pd.DataFrame:
    split_df = merged[merged["split"].eq(split) & merged["policy_name"].isin(POLICIES)].copy()
    value_wide = split_df.pivot(
        index=["split", "episode_id"],
        columns="policy_name",
        values="episode_final_after_tax_total_value",
    ).reset_index()
    value_wide = value_wide.rename(
        columns={
            PREFERRED_POLICY: "preferred_value",
            "hold_to_terminal": "hold_to_terminal_value",
            "sell_immediately": "sell_immediately_value",
            "sell_half_then_hold": "sell_half_then_hold_value",
        }
    )
    value_columns = [
        "preferred_value",
        "hold_to_terminal_value",
        "sell_immediately_value",
        "sell_half_then_hold_value",
    ]
    if value_wide[value_columns].isna().any().any():
        raise ValueError(f"Unpaired sector final-value rows for split={split}.")

    sharpe_long = split_df[["split", "episode_id", "policy_name", "EAAT_Sharpe", "TA_EAAT_Sharpe"]].copy()
    sharpe_long["policy_label"] = sharpe_long["policy_name"].map(POLICY_LABELS)
    sharpe_wide = sharpe_long.pivot(
        index=["split", "episode_id"],
        columns="policy_label",
        values=["EAAT_Sharpe", "TA_EAAT_Sharpe"],
    )
    sharpe_wide.columns = [
        f"{policy_label}_{metric}"
        for metric, policy_label in sharpe_wide.columns.to_flat_index()
    ]
    sharpe_wide = sharpe_wide.reset_index()

    episode_meta = (
        split_df[["split", "episode_id", "ticker", "sector", "industry"]]
        .drop_duplicates(["split", "episode_id"])
        .copy()
    )
    preferred = split_df[split_df["policy_name"].eq(PREFERRED_POLICY)][
        [
            "split",
            "episode_id",
            "total_tax_paid",
            "mean_effective_tax_rate_on_sales",
            "pct_episode_position_sold_short_term",
            "pct_episode_position_sold_long_term",
            "episode_cut_occurred",
            "num_discretionary_sales",
            "days_to_first_sale",
        ]
    ].rename(
        columns={
            "total_tax_paid": "preferred_tax_paid",
            "mean_effective_tax_rate_on_sales": "preferred_effective_tax_rate",
            "pct_episode_position_sold_short_term": "preferred_short_term_sold_fraction",
            "pct_episode_position_sold_long_term": "preferred_long_term_sold_fraction",
            "episode_cut_occurred": "preferred_cut_occurred",
            "num_discretionary_sales": "preferred_num_discretionary_sales",
            "days_to_first_sale": "preferred_days_to_first_sale",
        }
    )
    base = value_wide.merge(sharpe_wide, on=["split", "episode_id"], how="left", validate="one_to_one")
    base = base.merge(episode_meta, on=["split", "episode_id"], how="left", validate="one_to_one")
    base = base.merge(preferred, on=["split", "episode_id"], how="left", validate="one_to_one")
    base["preferred_no_cut"] = ~base["preferred_cut_occurred"]
    base["preferred_discretionary_sale"] = base["preferred_num_discretionary_sales"].fillna(0).gt(0)
    return base


def group_metrics(group: pd.DataFrame, *, split: str, sector: str, industry: str | None = None) -> dict[str, Any]:
    pref_minus_hold = group["preferred_value"] - group["hold_to_terminal_value"]
    pref_minus_sell = group["preferred_value"] - group["sell_immediately_value"]
    pref_minus_half = group["preferred_value"] - group["sell_half_then_hold_value"]
    pref_minus_hold_eaat = group["preferred_EAAT_Sharpe"] - group["hold_to_terminal_EAAT_Sharpe"]
    pref_minus_sell_eaat = group["preferred_EAAT_Sharpe"] - group["sell_immediately_EAAT_Sharpe"]
    pref_minus_half_eaat = group["preferred_EAAT_Sharpe"] - group["sell_half_then_hold_EAAT_Sharpe"]
    pref_minus_hold_ta = group["preferred_TA_EAAT_Sharpe"] - group["hold_to_terminal_TA_EAAT_Sharpe"]
    pref_minus_sell_ta = group["preferred_TA_EAAT_Sharpe"] - group["sell_immediately_TA_EAAT_Sharpe"]
    pref_minus_half_ta = group["preferred_TA_EAAT_Sharpe"] - group["sell_half_then_hold_TA_EAAT_Sharpe"]
    row: dict[str, Any] = {
        "split": split,
        "sector": sector,
        "num_episodes": int(len(group)),
        "small_group_warning": bool(len(group) < 30),
        "preferred_mean_final_after_tax_value": safe_mean(group["preferred_value"]),
        "hold_to_terminal_mean_final_after_tax_value": safe_mean(group["hold_to_terminal_value"]),
        "sell_immediately_mean_final_after_tax_value": safe_mean(group["sell_immediately_value"]),
        "sell_half_then_hold_mean_final_after_tax_value": safe_mean(group["sell_half_then_hold_value"]),
        "preferred_minus_hold_mean": safe_mean(pref_minus_hold),
        "preferred_minus_sell_immediately_mean": safe_mean(pref_minus_sell),
        "preferred_minus_sell_half_mean": safe_mean(pref_minus_half),
        "preferred_win_rate_vs_hold": safe_win_rate(pref_minus_hold),
        "preferred_win_rate_vs_sell_immediately": safe_win_rate(pref_minus_sell),
        "preferred_win_rate_vs_sell_half": safe_win_rate(pref_minus_half),
        "preferred_mean_tax_paid": safe_mean(group["preferred_tax_paid"]),
        "preferred_median_tax_paid": safe_median(group["preferred_tax_paid"]),
        "preferred_mean_effective_tax_rate": safe_mean(group["preferred_effective_tax_rate"]),
        "preferred_mean_short_term_sold_fraction": safe_mean(group["preferred_short_term_sold_fraction"]),
        "preferred_mean_long_term_sold_fraction": safe_mean(group["preferred_long_term_sold_fraction"]),
        "preferred_no_cut_pct": safe_mean(group["preferred_no_cut"].astype(float)),
        "preferred_discretionary_sale_pct": safe_mean(group["preferred_discretionary_sale"].astype(float)),
        "preferred_average_days_to_first_sale": safe_mean(group["preferred_days_to_first_sale"]),
        "preferred_median_days_to_first_sale": safe_median(group["preferred_days_to_first_sale"]),
        "preferred_median_EAAT_Sharpe": safe_median(group["preferred_EAAT_Sharpe"]),
        "hold_to_terminal_median_EAAT_Sharpe": safe_median(group["hold_to_terminal_EAAT_Sharpe"]),
        "sell_immediately_median_EAAT_Sharpe": safe_median(group["sell_immediately_EAAT_Sharpe"]),
        "sell_half_then_hold_median_EAAT_Sharpe": safe_median(group["sell_half_then_hold_EAAT_Sharpe"]),
        "preferred_median_TA_EAAT_Sharpe": safe_median(group["preferred_TA_EAAT_Sharpe"]),
        "hold_to_terminal_median_TA_EAAT_Sharpe": safe_median(group["hold_to_terminal_TA_EAAT_Sharpe"]),
        "sell_immediately_median_TA_EAAT_Sharpe": safe_median(group["sell_immediately_TA_EAAT_Sharpe"]),
        "sell_half_then_hold_median_TA_EAAT_Sharpe": safe_median(group["sell_half_then_hold_TA_EAAT_Sharpe"]),
        "preferred_minus_hold_median_EAAT_Sharpe": safe_median(pref_minus_hold_eaat),
        "preferred_EAAT_Sharpe_win_rate_vs_hold": safe_win_rate(pref_minus_hold_eaat),
        "preferred_minus_hold_median_TA_EAAT_Sharpe": safe_median(pref_minus_hold_ta),
        "preferred_TA_EAAT_Sharpe_win_rate_vs_hold": safe_win_rate(pref_minus_hold_ta),
        "preferred_minus_sell_immediately_median_EAAT_Sharpe": safe_median(pref_minus_sell_eaat),
        "preferred_minus_sell_half_median_EAAT_Sharpe": safe_median(pref_minus_half_eaat),
        "preferred_minus_sell_immediately_median_TA_EAAT_Sharpe": safe_median(pref_minus_sell_ta),
        "preferred_minus_sell_half_median_TA_EAAT_Sharpe": safe_median(pref_minus_half_ta),
    }
    if industry is not None:
        row = {"split": split, "sector": sector, "industry": industry, **{k: v for k, v in row.items() if k not in {"split", "sector"}}}
    return row


def build_group_table(
    merged: pd.DataFrame,
    *,
    splits: list[str],
    group_columns: list[str],
    min_episodes_per_group: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        base = paired_base_for_split(merged, split)
        for group_key, group in base.groupby(group_columns, sort=False, observed=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            if min_episodes_per_group is not None and len(group) < min_episodes_per_group:
                continue
            sector = str(group_key[0])
            industry = str(group_key[1]) if len(group_key) > 1 else None
            rows.append(group_metrics(group, split=split, sector=sector, industry=industry))
    out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return out


def validate_output(df: pd.DataFrame, *, name: str, is_industry: bool = False) -> None:
    if df.empty:
        raise ValueError(f"Generated empty sector analysis table: {name}")
    required = REQUIRED_OUTPUT_COLUMNS.copy()
    if is_industry:
        required = ["split", "sector", "industry", *[c for c in required if c not in {"split", "sector"}]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required column(s): " + ", ".join(missing))
    bad = sorted(set(df.columns) & LEGACY_SHARPE_COLUMNS)
    if bad:
        raise ValueError(f"{name} contains forbidden legacy Sharpe column(s): " + ", ".join(bad))
    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"{name} contains infinite numeric values.")


def sector_plot_order(sector_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = sector_df[sector_df["split"].eq("test")].copy()
    if plot_df.empty:
        raise ValueError("No test rows available for sector plots.")
    return plot_df.sort_values("preferred_minus_hold_mean", ascending=True)


def plot_sector_bar(
    sector_df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    manifest: Manifest,
    description: str,
) -> None:
    plot_df = sector_plot_order(sector_df)
    colors = np.where(plot_df[metric].ge(0), "#3b6ea8", "#8a4f3d")
    plt.figure(figsize=(11, 5.4))
    plt.bar(plot_df["sector"], plot_df[metric], color=colors)
    if plot_df[metric].min(skipna=True) < 0 < plot_df[metric].max(skipna=True):
        plt.axhline(0, color="black", linewidth=1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=40, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_plot(path, manifest, description)


def plot_sector_sharpe(
    sector_df: pd.DataFrame,
    *,
    metric_suffix: str,
    ylabel: str,
    title: str,
    path: Path,
    manifest: Manifest,
    description: str,
) -> None:
    plot_df = sector_plot_order(sector_df)
    x = np.arange(len(plot_df))
    width = 0.38
    plt.figure(figsize=(11.5, 5.6))
    plt.bar(
        x - width / 2,
        plot_df[f"preferred_median_{metric_suffix}"],
        width,
        label="preferred",
        color="#3b6ea8",
    )
    plt.bar(
        x + width / 2,
        plot_df[f"hold_to_terminal_median_{metric_suffix}"],
        width,
        label="hold_to_terminal",
        color="#6c7a40",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, plot_df["sector"], rotation=40, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_plot(path, manifest, description)


def write_notes(
    path: Path,
    *,
    metadata_path: Path,
    ticker_extraction_method: str,
    metadata_summary: dict[str, Any],
    sector_df: pd.DataFrame,
    min_episodes_per_group: int,
    manifest: Manifest,
) -> None:
    unknown_exists = bool(sector_df["sector"].eq("Unknown").any())
    lines = [
        "Sector analysis notes",
        "metadata_source: yfinance.Ticker.info",
        f"metadata_cache_file: {relative_path(metadata_path)}",
        f"ticker_extraction_method: {ticker_extraction_method}",
        f"number_of_tickers_found: {metadata_summary['num_tickers_found']}",
        f"number_with_sector: {metadata_summary['num_with_sector']}",
        f"number_missing_sector: {metadata_summary['num_missing_sector']}",
        f"number_failed: {metadata_summary['num_failed']}",
        f"unknown_sector_exists: {unknown_exists}",
        f"industry_min_episodes_per_group: {min_episodes_per_group}",
        "sector_analysis_scope: diagnostic interpretation only.",
        "final_after_tax_value_primary_metric: True",
        "EAAT_and_TA_EAAT_Sharpe_secondary_risk_adjusted_diagnostics: True",
        "Sharpe_difference_diagnostics_use_medians_not_means: True",
        "policy_selection_changed_based_on_sector_results: False",
        f"preferred_policy: {PREFERRED_POLICY}",
        "benchmarks: hold_to_terminal; sell_immediately; sell_half_then_hold",
        "interpretation_focus: sectors where preferred DQN helps or fails, and whether patterns align with active liquidation, tax timing, or no-cut behavior.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    manifest.add(path, "txt", "Sector analysis notes and provenance.")


def write_manifest(path: Path, manifest: Manifest) -> None:
    manifest.add(path, "json", "Sector analysis manifest.")
    path.write_text(json.dumps(manifest.rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sector-level policy diagnostics for the frozen final run."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-refresh-metadata", action="store_true")
    parser.add_argument("--metadata-max-workers", type=int, default=8)
    parser.add_argument("--min-episodes-per-group", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    metadata_path = resolve_project_path(args.metadata)
    config, run_dir = verify_run(config_path, output_dir)
    del config

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest()

    episode_df = load_episode_metrics(run_dir)
    validate_policy_rows(episode_df, PRIMARY_SPLITS)
    tickers, extraction_notes = extract_tickers_from_artifacts(run_dir)
    ticker_extraction_method = "; ".join(
        note.split(": ", 1)[1]
        if note.startswith("ticker_extraction_method: ")
        else note
        for note in extraction_notes
    )
    metadata, metadata_summary, _ = ensure_metadata(
        run_dir=run_dir,
        metadata_path=metadata_path,
        force_refresh=args.force_refresh_metadata,
        max_workers=max(1, int(args.metadata_max_workers)),
    )
    manifest.add(metadata_path, "csv", "Reusable ticker sector metadata cache.")
    metadata_notes_path = metadata_path.with_name(metadata_path.stem + "_notes.txt")
    if metadata_notes_path.exists():
        manifest.add(metadata_notes_path, "txt", "Ticker sector metadata notes.")
    joined = merge_metadata(episode_df, metadata)
    sharpe = load_episode_sharpe(output_dir, PRIMARY_SPLITS)
    merged = merge_sharpe(joined, sharpe, PRIMARY_SPLITS)

    sector_df = build_group_table(
        merged,
        splits=PRIMARY_SPLITS,
        group_columns=["sector"],
    )
    validate_output(sector_df, name="sector_policy_performance")
    sector_df = sector_df.sort_values(["split", "preferred_minus_hold_mean", "sector"])
    save_table(
        sector_df,
        output_dir / "sector_policy_performance.csv",
        output_dir / "sector_policy_performance.md",
        manifest,
        "Sector-level preferred policy versus benchmark diagnostics.",
    )

    industry_df = build_group_table(
        merged,
        splits=PRIMARY_SPLITS,
        group_columns=["sector", "industry"],
        min_episodes_per_group=max(1, int(args.min_episodes_per_group)),
    )
    if not industry_df.empty:
        validate_output(
            industry_df,
            name="sector_industry_policy_performance",
            is_industry=True,
        )
        industry_df = industry_df.sort_values(
            ["split", "sector", "preferred_minus_hold_mean", "industry"]
        )
        save_table(
            industry_df,
            output_dir / "sector_industry_policy_performance.csv",
            output_dir / "sector_industry_policy_performance.md",
            manifest,
            "Sector-industry preferred policy versus benchmark diagnostics.",
        )

    plot_sector_bar(
        sector_df,
        metric="preferred_minus_hold_mean",
        ylabel="preferred minus hold mean final value",
        title="Preferred minus hold by sector - test",
        path=plots_dir / "sector_preferred_minus_hold_test.png",
        manifest=manifest,
        description="Preferred minus hold mean final after-tax value by sector for test split.",
    )
    plot_sector_bar(
        sector_df,
        metric="preferred_win_rate_vs_hold",
        ylabel="preferred win rate vs hold",
        title="Preferred win rate vs hold by sector - test",
        path=plots_dir / "sector_preferred_win_rate_vs_hold_test.png",
        manifest=manifest,
        description="Preferred win rate versus hold_to_terminal by sector for test split.",
    )
    plot_sector_bar(
        sector_df,
        metric="preferred_no_cut_pct",
        ylabel="preferred no-cut episode share",
        title="Preferred no-cut share by sector - test",
        path=plots_dir / "sector_no_cut_pct_test.png",
        manifest=manifest,
        description="Preferred policy no-cut episode share by sector for test split.",
    )
    plot_sector_bar(
        sector_df,
        metric="preferred_mean_short_term_sold_fraction",
        ylabel="preferred mean short-term sold fraction",
        title="Preferred short-term sold fraction by sector - test",
        path=plots_dir / "sector_short_term_sold_fraction_test.png",
        manifest=manifest,
        description="Preferred policy short-term sold fraction by sector for test split.",
    )
    plot_sector_sharpe(
        sector_df,
        metric_suffix="EAAT_Sharpe",
        ylabel="median EAAT Sharpe",
        title="Median EAAT Sharpe by sector - test",
        path=plots_dir / "sector_median_eaat_sharpe_test.png",
        manifest=manifest,
        description="Preferred and hold_to_terminal median EAAT Sharpe by sector for test split.",
    )
    plot_sector_sharpe(
        sector_df,
        metric_suffix="TA_EAAT_Sharpe",
        ylabel="median TA-EAAT Sharpe",
        title="Median TA-EAAT Sharpe by sector - test",
        path=plots_dir / "sector_median_ta_eaat_sharpe_test.png",
        manifest=manifest,
        description="Preferred and hold_to_terminal median TA-EAAT Sharpe by sector for test split.",
    )

    write_notes(
        output_dir / "sector_analysis_notes.txt",
        metadata_path=metadata_path,
        ticker_extraction_method=ticker_extraction_method,
        metadata_summary=metadata_summary,
        sector_df=sector_df,
        min_episodes_per_group=max(1, int(args.min_episodes_per_group)),
        manifest=manifest,
    )
    write_manifest(output_dir / "sector_analysis_manifest.json", manifest)
    print(f"tickers used: {len(tickers)}")
    print(f"sectors in table: {sector_df['sector'].nunique()}")
    print(f"output directory: {relative_path(output_dir)}")


if __name__ == "__main__":
    main()
