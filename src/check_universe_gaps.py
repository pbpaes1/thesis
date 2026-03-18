import argparse
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
]

VALUE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check missing values and missing business days in universe.parquet"
    )
    parser.add_argument(
        "--input",
        default="data/raw/universe.parquet",
        help="Path to input parquet file (default: data/raw/universe.parquet)",
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date inclusive (default: 2010-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-01-01",
        help="End date exclusive (default: 2026-01-01)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/quality",
        help="Directory for output reports (default: data/quality)",
    )
    parser.add_argument(
        "--calendar",
        default="XNYS",
        help="Exchange calendar code for trading sessions (default: XNYS)",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def load_and_filter(input_path: Path, start: str, end: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    validate_columns(df)

    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    filtered = df[(df["date"] >= start_ts) & (df["date"] < end_ts)].copy()
    filtered = filtered.sort_values(["ticker", "date"]).reset_index(drop=True)
    return filtered


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    report = df.groupby("ticker", as_index=True)[VALUE_COLUMNS].agg(
        lambda s: s.isna().sum()
    )
    report = report.rename(columns={col: f"missing_{col}" for col in VALUE_COLUMNS})
    report = report.reset_index()
    report["missing_total"] = report[[f"missing_{col}" for col in VALUE_COLUMNS]].sum(axis=1)
    report = report.sort_values(["missing_total", "ticker"], ascending=[False, True])
    return report


def _calendar_sessions(calendar_name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return exchange trading sessions between start and end (inclusive)."""
    cal = xcals.get_calendar(calendar_name)
    sessions = cal.sessions_in_range(start, end)
    sessions = pd.DatetimeIndex(sessions)
    if sessions.tz is not None:
        sessions = sessions.tz_convert(None)
    return sessions.normalize()


def missing_business_days_reports(
    df: pd.DataFrame,
    all_sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []

    for ticker, g in df.groupby("ticker", sort=True):
        actual_dates = pd.DatetimeIndex(g["date"].dropna().unique()).sort_values().normalize()
        if actual_dates.empty:
            summary_rows.append(
                {
                    "ticker": ticker,
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                    "observed_days": 0,
                    "expected_trading_days": 0,
                    "missing_trading_days": 0,
                }
            )
            continue

        first_date = actual_dates.min()
        last_date = actual_dates.max()
        expected = all_sessions[(all_sessions >= first_date) & (all_sessions <= last_date)]
        missing = expected.difference(actual_dates)

        summary_rows.append(
            {
                "ticker": ticker,
                "first_date": first_date,
                "last_date": last_date,
                "observed_days": len(actual_dates),
                "expected_trading_days": len(expected),
                "missing_trading_days": len(missing),
            }
        )

        for d in missing:
            detail_rows.append({"ticker": ticker, "missing_date": d})

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["missing_trading_days", "ticker"], ascending=[False, True]
    )
    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["ticker", "missing_date"]).reset_index(drop=True)

    return summary_df, detail_df


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(input_path, args.start, args.end)

    if df.empty:
        print("No data found in the selected date range.")
        return

    # Precompute exchange sessions once for performance.
    all_sessions = _calendar_sessions(
        args.calendar,
        pd.Timestamp(args.start),
        pd.Timestamp(args.end) - pd.Timedelta(days=1),
    )

    missing_values = missing_values_report(df)
    gap_summary, gap_details = missing_business_days_reports(df, all_sessions)

    filtered_out = output_dir / "universe_filtered_2010_2026.parquet"
    missing_values_out = output_dir / "missing_values_by_ticker.csv"
    gaps_summary_out = output_dir / "missing_trading_days_summary.csv"
    gaps_detail_out = output_dir / "missing_trading_days_detail.csv"

    df.to_parquet(filtered_out, index=False)
    missing_values.to_csv(missing_values_out, index=False)
    gap_summary.to_csv(gaps_summary_out, index=False)
    gap_details.to_csv(gaps_detail_out, index=False)

    print("Done.")
    print(f"Filtered rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Saved: {filtered_out}")
    print(f"Saved: {missing_values_out}")
    print(f"Saved: {gaps_summary_out}")
    print(f"Saved: {gaps_detail_out}")
    print(f"Calendar used: {args.calendar}")


if __name__ == "__main__":
    main()
