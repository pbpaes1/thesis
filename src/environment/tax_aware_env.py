"""Tax-aware RL environment contract.

Environment contract:
- Input dataset: episode-level parquet.
- One episode: one `episode_id`.
- One step: one row in chronological order within an episode.
- Observation: vector built from externally supplied `state_columns`, expected to
  be loaded upstream from the frozen schema artifact
  `allowed_state_columns_v1.json`.
- Actions: discrete sell fractions of the original position (not remaining position).
- Required rollout structure: index columns (`episode_id`, `date`) plus
  bookkeeping columns (`tax_transition_date`, `unrealized_gains_pct`).
- Bookkeeping: sold fraction, remaining fraction, cumulative realized pre-tax PnL,
  cumulative realized after-tax PnL, and Reward A after-tax value components.
- Episode end (`done`): final row reached or 100% liquidated.
- This skeleton remains generic and does not load the frozen JSON artifact itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.accounting.brazil_v1 import after_tax_lot_value, calculate_sale, gross_lot_value
from src.config.tax_profiles import resolve_economic_scenario_from_config


Observation = npt.NDArray[np.float32]
InfoDict = dict[str, Any]
ResetResult = tuple[Observation, InfoDict]
StepResult = tuple[Observation, float, bool, bool, InfoDict]

REWARD_A_VERSION = "A_after_tax_total_value_change"
REWARD_C_LITE_VERSION = "C_lite_after_tax_value_change_minus_cooldown_penalty"
REWARD_C_LITE_V2_VERSION = (
    "C_lite_v2_after_tax_value_change_minus_transaction_and_cooldown_penalty"
)
BRAZIL_BASE_REWARD_VERSION = "brazil_v1_total_after_tax_wealth_change"
BRAZIL_C_LITE_REWARD_VERSION = (
    "brazil_v1_total_after_tax_wealth_change_minus_cooldown_penalty"
)
SUPPORTED_REWARD_VERSIONS = {
    REWARD_A_VERSION,
    REWARD_C_LITE_VERSION,
    REWARD_C_LITE_V2_VERSION,
    BRAZIL_BASE_REWARD_VERSION,
    BRAZIL_C_LITE_REWARD_VERSION,
}
COOLDOWN_REWARD_VERSIONS = {
    REWARD_C_LITE_VERSION, REWARD_C_LITE_V2_VERSION, BRAZIL_C_LITE_REWARD_VERSION,
}


@dataclass
class _BrazilCashLot:
    principal: float
    deposit_date: pd.Timestamp
    market_day_age: int = 0


class TaxAwareEnv:
    """Plain Python skeleton for a tax-aware optimal stopping / liquidation environment.

    Observation schema is external to this class and passed in through
    `state_columns` (typically loaded from `allowed_state_columns_v1.json`).
    """

    # Discrete actions map to sell fractions of the original position.
    DEFAULT_ACTION_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)

    # Minimum columns needed for episode traversal in rollout.
    REQUIRED_INDEX_COLUMNS: tuple[str, ...] = ("episode_id", "date")

    # Minimum bookkeeping inputs expected from the episode dataset.
    REQUIRED_BOOKKEEPING_COLUMNS: tuple[str, ...] = (
        "tax_transition_date",
        "unrealized_gains_pct",
    )

    # Default individual tax profile placeholder.
    DEFAULT_TAX_CONFIG: dict[str, Any] = {
        "profile_name": "unspecified_individual",
        "short_term_rate": None,
        "long_term_rate": None,
        "niit_rate": 0.0,
        "apply_niit": False,
    }

    # Optional RNG seed placeholder.
    DEFAULT_SEED: int | None = None

    def __init__(
        self,
        parquet_path: str | Path,
        state_columns: Sequence[str],
        action_fractions: Sequence[float] | None = None,
        tax_config: Mapping[str, Any] | None = None,
        reward_config: Mapping[str, Any] | None = None,
        seed: int | None = None,
        economic_scenario: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize lightweight configuration and runtime placeholders.

        This constructor intentionally avoids full environment mechanics.
        It only stores configuration and initializes internal state scaffolding.
        `state_columns` should be loaded outside the class from the frozen
        state schema artifact (`allowed_state_columns_v1.json`) and passed in.
        """
        self.parquet_path = Path(parquet_path)
        self.state_columns = list(state_columns)
        self.action_fractions = tuple(
            action_fractions if action_fractions is not None else self.DEFAULT_ACTION_FRACTIONS
        )
        self.tax_config = dict(
            tax_config if tax_config is not None else self.DEFAULT_TAX_CONFIG
        )
        self.reward_config = dict(reward_config if reward_config is not None else {})
        self.economic_scenario = resolve_economic_scenario_from_config(
            {"economic_scenario": economic_scenario}
        ) if economic_scenario is not None else None
        self._brazil_mode = self.economic_scenario is not None
        self.reward_version = str(
            self.reward_config.get("version", REWARD_A_VERSION)
        )
        if self.reward_version not in SUPPORTED_REWARD_VERSIONS:
            raise ValueError(
                f"Unsupported reward version {self.reward_version!r}. "
                f"Supported versions: {sorted(SUPPORTED_REWARD_VERSIONS)}."
            )
        brazil_reward = self.reward_version in {
            BRAZIL_BASE_REWARD_VERSION, BRAZIL_C_LITE_REWARD_VERSION,
        }
        if brazil_reward != self._brazil_mode:
            raise ValueError(
                "Brazil scenario and Brazil wealth-based reward identifiers must be selected together."
            )
        self.base_reward_version = str(
            self.reward_config.get("base_reward_version", REWARD_A_VERSION)
        )
        if self.reward_version in COOLDOWN_REWARD_VERSIONS:
            required_base = (
                BRAZIL_BASE_REWARD_VERSION
                if self._brazil_mode else REWARD_A_VERSION
            )
            if self.base_reward_version != required_base:
                raise ValueError(
                    "Reward C-lite rewards require base_reward_version="
                    f"{required_base!r}, got {self.base_reward_version!r}."
                )
        if self.reward_version == BRAZIL_BASE_REWARD_VERSION and self.base_reward_version != BRAZIL_BASE_REWARD_VERSION:
            raise ValueError("Brazil base reward requires matching base_reward_version.")

        raw_transaction_config = self.reward_config.get("transaction_penalty", {})
        if raw_transaction_config is None:
            raw_transaction_config = {}
        if not isinstance(raw_transaction_config, Mapping):
            raise ValueError(
                "reward.transaction_penalty must be a mapping when provided."
            )
        self.transaction_penalty_config = dict(raw_transaction_config)
        default_transaction_enabled = self.reward_version == REWARD_C_LITE_V2_VERSION
        self._transaction_penalty_enabled = bool(
            self.transaction_penalty_config.get(
                "enabled",
                self.reward_config.get(
                    "use_transaction_penalty",
                    default_transaction_enabled,
                ),
            )
        )
        self._lambda_transaction = float(
            self.transaction_penalty_config.get("lambda_transaction", 0.0)
        )
        self._scale_transaction_by_executed_fraction = bool(
            self.transaction_penalty_config.get("scale_by_executed_fraction", True)
        )
        self._apply_transaction_to_first_sale = bool(
            self.transaction_penalty_config.get("apply_to_first_sale", True)
        )
        self._apply_transaction_to_automatic_terminal_liquidation = bool(
            self.transaction_penalty_config.get(
                "apply_to_automatic_terminal_liquidation",
                False,
            )
        )

        raw_cooldown_config = self.reward_config.get("cooldown_penalty", {})
        if raw_cooldown_config is None:
            raw_cooldown_config = {}
        if not isinstance(raw_cooldown_config, Mapping):
            raise ValueError("reward.cooldown_penalty must be a mapping when provided.")
        self.cooldown_penalty_config = dict(raw_cooldown_config)
        default_cooldown_enabled = self.reward_version in COOLDOWN_REWARD_VERSIONS
        self._cooldown_penalty_enabled = bool(
            self.cooldown_penalty_config.get(
                "enabled",
                self.reward_config.get(
                    "use_cooldown_penalty",
                    default_cooldown_enabled,
                ),
            )
        )
        self._lambda_cooldown = float(
            self.cooldown_penalty_config.get("lambda_cooldown", 0.0)
        )
        self._cooldown_days = int(
            self.cooldown_penalty_config.get("cooldown_days", 20)
        )
        self._scale_cooldown_by_executed_fraction = bool(
            self.cooldown_penalty_config.get("scale_by_executed_fraction", True)
        )
        self._penalize_first_sale = bool(
            self.cooldown_penalty_config.get("penalize_first_sale", False)
        )
        self._apply_cooldown_to_automatic_terminal_liquidation = bool(
            self.cooldown_penalty_config.get(
                "apply_to_automatic_terminal_liquidation",
                False,
            )
        )
        if self.reward_version in COOLDOWN_REWARD_VERSIONS:
            if not self._cooldown_penalty_enabled:
                raise ValueError("Reward C-lite requires cooldown_penalty.enabled=true.")
            if self._lambda_cooldown < 0.0:
                raise ValueError("lambda_cooldown must be non-negative.")
            if self._cooldown_days <= 0:
                raise ValueError("cooldown_days must be positive.")
            if self._penalize_first_sale:
                raise ValueError("Reward C-lite does not penalize the first sale.")
            if self._apply_cooldown_to_automatic_terminal_liquidation:
                raise ValueError(
                    "Reward C-lite does not apply cooldown penalties to "
                    "automatic terminal liquidation."
                )
        if self.reward_version == REWARD_C_LITE_V2_VERSION:
            if not self._transaction_penalty_enabled:
                raise ValueError(
                    "Reward C-lite v2 requires transaction_penalty.enabled=true."
                )
            if self._lambda_transaction < 0.0:
                raise ValueError("lambda_transaction must be non-negative.")
            if not self._apply_transaction_to_first_sale:
                raise ValueError(
                    "Reward C-lite v2 requires transaction penalties on first sales."
                )
            if self._apply_transaction_to_automatic_terminal_liquidation:
                raise ValueError(
                    "Reward C-lite v2 does not apply transaction penalties to "
                    "automatic terminal liquidation."
                )
        self.seed = seed if seed is not None else self.DEFAULT_SEED
        self._rng = np.random.default_rng(self.seed)

        # TODO(next): validate parquet path existence and schema compatibility.
        # TODO(next): validate REQUIRED_INDEX_COLUMNS and REQUIRED_BOOKKEEPING_COLUMNS.
        # TODO(next): validate that state_columns come from frozen v1 schema.
        # TODO(next): validate that all supplied state_columns exist in parquet.
        # TODO(next): validate action fractions are monotone and within [0, 1].

        # Loaded full parquet dataframe (all episodes).
        self._df: pd.DataFrame | None = None
        # Mapping from episode_id -> row index positions in chronological order.
        self._episode_index: dict[str, pd.Index] = {}
        # Currently selected episode_id after reset.
        self._current_episode_id: str | None = None
        # Current episode slice dataframe used by reset/step rollouts.
        self._current_episode_df: pd.DataFrame | None = None
        # Pointer to current row inside current episode dataframe.
        self._current_row_ptr: int = 0

        # Fraction bookkeeping (relative to original position).
        self._sold_fraction: float = 0.0
        self._remaining_fraction: float = 1.0

        # Cumulative realized PnL bookkeeping placeholders.
        self._cum_realized_pre_tax_pnl: float = 0.0
        self._cum_realized_after_tax_pnl: float = 0.0

        # Reward A state: total after-tax value = realized after-tax PnL
        # plus hypothetical after-tax liquidation value of remaining inventory.
        self._prev_after_tax_total_value: float = 0.0
        self._after_tax_total_value: float = 0.0
        self._after_tax_liquidation_value_remaining: float = 0.0

        # Reward C-lite cooldown state. Automatic terminal liquidation does not
        # update these fields because it is not a discretionary agent sale.
        self._last_sale_date: pd.Timestamp | None = None
        self._sale_count: int = 0

        # Used only by the explicit Brazil path. The U.S. reset/step logic is
        # dispatched unchanged below.
        self._brazil_lots: list[_BrazilCashLot] = []
        self._brazil_done = False
        self._brazil_previous_wealth = 0.0
        self._brazil_initial_wealth = 0.0
        self._brazil_cash_interest_cumulative = 0.0
        self._brazil_fixed_income_tax_cumulative = 0.0
        self._brazil_equity_tax_cumulative = 0.0
        self._brazil_gross_proceeds_cumulative = 0.0
        self._brazil_cash_principal_cumulative = 0.0
        self._brazil_mandatory_sale_count = 0
        self._brazil_first_sale_date: pd.Timestamp | None = None
        self._brazil_previous_observation_date: pd.Timestamp | None = None

    def _resolve_effective_tax_rate(
        self,
        tax_regime: str,
        *,
        require_rates: bool,
    ) -> float:
        """Resolve effective tax rate for the active individual tax profile.

        Effective-rate convention used in this thesis environment:
        - short-term: short_term_rate
        - long-term: long_term_rate + (niit_rate if apply_niit else 0)
        """
        def _as_float(value: Any, key: str) -> float:
            if value is None:
                raise ValueError(f"Tax config key '{key}' is required but missing/None.")
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Tax config key '{key}' must be numeric, got value={value!r}."
                ) from exc

        short_rate_raw = self.tax_config.get("short_term_rate")
        long_rate_raw = self.tax_config.get("long_term_rate")

        if short_rate_raw is None or long_rate_raw is None:
            if require_rates:
                raise ValueError(
                    "Taxable gain valuation requires both 'short_term_rate' "
                    "and 'long_term_rate' in tax_config."
                )
            return 0.0

        short_rate = _as_float(short_rate_raw, "short_term_rate")
        long_rate = _as_float(long_rate_raw, "long_term_rate")
        niit_rate = float(self.tax_config.get("niit_rate", 0.0) or 0.0)
        apply_niit = bool(self.tax_config.get("apply_niit", False))

        effective_short_term_rate = short_rate
        effective_long_term_rate = long_rate + (
            niit_rate if apply_niit else 0.0
        )

        if tax_regime == "short_term":
            return float(effective_short_term_rate)
        if tax_regime == "long_term":
            return float(effective_long_term_rate)
        raise ValueError(f"Unsupported tax_regime '{tax_regime}'.")

    def _classify_tax_regime(self, row: pd.Series) -> str:
        """Classify row-level tax regime from date vs tax transition date."""
        row_date = pd.to_datetime(row["date"], errors="coerce")
        tax_transition_date = pd.to_datetime(
            row["tax_transition_date"], errors="coerce"
        )
        if pd.isna(row_date) or pd.isna(tax_transition_date):
            raise ValueError(
                "Cannot classify tax regime due to invalid 'date' or "
                "'tax_transition_date' in row."
            )
        return "short_term" if row_date < tax_transition_date else "long_term"

    def _get_full_position_pnl(self, row: pd.Series) -> float:
        """Read full-position unrealized PnL from the current episode row."""
        full_position_pnl = float(row["unrealized_gains_pct"])
        if np.isnan(full_position_pnl):
            raise ValueError(
                "Column 'unrealized_gains_pct' contains NaN in current row."
            )
        return full_position_pnl

    def _parse_step_date(self, value: Any, *, context: str) -> pd.Timestamp:
        """Parse an environment row date for cooldown bookkeeping."""
        step_date = pd.to_datetime(value, errors="coerce")
        if pd.isna(step_date):
            raise ValueError(
                f"Cannot compute sale cooldown state: invalid date in {context}: "
                f"{value!r}."
            )
        return pd.Timestamp(step_date)

    def _compute_cooldown_penalty(
        self,
        *,
        executable_fraction: float,
        current_sale_date: pd.Timestamp | None,
    ) -> tuple[float, bool, int | None, bool]:
        """Compute the Reward C-lite cooldown penalty before sale-state update."""
        previous_sale_exists = bool(
            self._sale_count > 0 and self._last_sale_date is not None
        )
        days_since_last_sale: int | None = None
        cooldown_penalty = 0.0
        cooldown_penalty_applied = False

        if executable_fraction <= 0.0:
            return (
                cooldown_penalty,
                cooldown_penalty_applied,
                days_since_last_sale,
                previous_sale_exists,
            )
        if current_sale_date is None:
            raise ValueError(
                "Internal error: executable sale requires a parsed current sale date."
            )
        if previous_sale_exists:
            delta_days = int((current_sale_date - self._last_sale_date).days)
            if delta_days < 0:
                raise ValueError(
                    "Sale dates must be non-decreasing within an episode for "
                    "cooldown penalty computation. "
                    f"last_sale_date={self._last_sale_date}, "
                    f"current_sale_date={current_sale_date}."
                )
            days_since_last_sale = delta_days

        if (
            self.reward_version in COOLDOWN_REWARD_VERSIONS
            and self._cooldown_penalty_enabled
            and previous_sale_exists
            and days_since_last_sale is not None
            and days_since_last_sale < self._cooldown_days
        ):
            scale = (
                float(executable_fraction)
                if self._scale_cooldown_by_executed_fraction
                else 1.0
            )
            cooldown_penalty = float(
                self._lambda_cooldown
                * scale
                * max(0, self._cooldown_days - days_since_last_sale)
                / self._cooldown_days
            )
            cooldown_penalty_applied = bool(cooldown_penalty > 0.0)

        return (
            cooldown_penalty,
            cooldown_penalty_applied,
            days_since_last_sale,
            previous_sale_exists,
        )

    def _compute_transaction_penalty(
        self,
        *,
        executable_fraction: float,
        is_automatic_terminal_liquidation: bool,
    ) -> tuple[float, bool]:
        """Compute the Reward C-lite v2 discretionary sale transaction penalty."""
        transaction_penalty = 0.0
        transaction_penalty_applied = False

        if (
            self.reward_version != REWARD_C_LITE_V2_VERSION
            or not self._transaction_penalty_enabled
            or executable_fraction <= 0.0
        ):
            return transaction_penalty, transaction_penalty_applied
        if is_automatic_terminal_liquidation:
            if not self._apply_transaction_to_automatic_terminal_liquidation:
                return transaction_penalty, transaction_penalty_applied

        scale = (
            float(executable_fraction)
            if self._scale_transaction_by_executed_fraction
            else 1.0
        )
        transaction_penalty = float(self._lambda_transaction * scale)
        transaction_penalty_applied = bool(transaction_penalty > 0.0)
        return transaction_penalty, transaction_penalty_applied

    def _compute_after_tax_pnl_increment(
        self,
        pre_tax_increment: float,
        tax_rate: float,
    ) -> tuple[float, float]:
        """Return `(tax_paid, after_tax_increment)` without tax credits on losses."""
        if pre_tax_increment > 0.0:
            if np.isnan(tax_rate):
                raise ValueError(
                    "Positive pre-tax gain requires a valid applicable tax rate."
                )
            tax_paid = float(pre_tax_increment * tax_rate)
        else:
            tax_paid = 0.0
        after_tax_increment = float(pre_tax_increment - tax_paid)
        return tax_paid, after_tax_increment

    def _compute_after_tax_liquidation_value(
        self,
        *,
        remaining_fraction: float,
        full_position_pnl: float,
        tax_regime: str,
    ) -> tuple[float, float, float, float]:
        """Value remaining inventory under a hypothetical after-tax liquidation.

        Returns:
            `(after_tax_value, tax_rate, pre_tax_value, tax_drag)`.
        """
        remaining_pre_tax_value = float(remaining_fraction * full_position_pnl)
        require_rates = bool(remaining_pre_tax_value > 0.0)
        tax_rate = self._resolve_effective_tax_rate(
            tax_regime,
            require_rates=require_rates,
        )
        tax_drag, after_tax_value = self._compute_after_tax_pnl_increment(
            remaining_pre_tax_value,
            tax_rate,
        )
        return after_tax_value, tax_rate, remaining_pre_tax_value, tax_drag

    def _load_episode_index(self) -> dict[str, pd.Index]:
        """Load parquet data and build episode_id -> row-index mapping.

        Returns:
            A mapping where each key is an `episode_id` and the value identifies
            row indices in chronological order for that episode.
        """
        if self._df is not None and self._episode_index:
            return self._episode_index

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        missing_index_cols = [
            col for col in self.REQUIRED_INDEX_COLUMNS if col not in df.columns
        ]
        if missing_index_cols:
            raise ValueError(
                "Parquet is missing required index columns: "
                f"{missing_index_cols}"
            )

        bookkeeping_columns = (
            ("adj_close", "simulated_purchase_price")
            if self._brazil_mode else self.REQUIRED_BOOKKEEPING_COLUMNS
        )
        missing_bookkeeping_cols = [
            col for col in bookkeeping_columns if col not in df.columns
        ]
        if missing_bookkeeping_cols:
            raise ValueError(
                "Parquet is missing required bookkeeping columns: "
                f"{missing_bookkeeping_cols}"
            )

        missing_state_cols = [col for col in self.state_columns if col not in df.columns]
        if missing_state_cols:
            raise ValueError(
                "Supplied state_columns are missing in parquet: "
                f"{missing_state_cols}"
            )

        if df["episode_id"].isna().any():
            raise ValueError("Column 'episode_id' contains missing values.")
        df = df.copy()
        df["episode_id"] = df["episode_id"].astype(str)

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError(
                "Column 'date' contains invalid or missing values after datetime conversion."
            )

        df = df.sort_values(["episode_id", "date"], kind="stable").reset_index(drop=True)

        if self._brazil_mode:
            for column in ("adj_close", "simulated_purchase_price"):
                df[column] = pd.to_numeric(df[column], errors="coerce")
                if not np.isfinite(df[column].to_numpy(dtype=float)).all() or (df[column] <= 0).any():
                    raise ValueError(f"Brazil episode column {column!r} must contain finite positive prices.")
            if df.duplicated(["episode_id", "date"]).any():
                raise ValueError("Brazil episode dates must be strictly increasing within each episode.")
            if (df.groupby("episode_id")["simulated_purchase_price"].nunique() != 1).any():
                raise ValueError("Brazil episode simulated_purchase_price must be constant within each episode.")

        grouped_indices = df.groupby("episode_id", sort=True).indices
        episode_index = {
            episode_id: pd.Index(row_positions)
            for episode_id, row_positions in grouped_indices.items()
        }
        if not episode_index:
            raise ValueError(
                "No episodes found in parquet after indexing by 'episode_id'."
            )

        self._df = df
        self._episode_index = episode_index
        return self._episode_index

    def reset(self, episode_id: str | None = None) -> ResetResult:
        """Start a new episode rollout and return `(observation, info)`.

        Args:
            episode_id: Optional explicit episode id. If None, selection logic
                defaults to deterministic first-episode selection.

        Returns:
            Tuple of:
            - initial observation vector using the frozen state column schema
            - reset info dictionary with key bookkeeping fields
        """
        if self._brazil_mode:
            return self._reset_brazil(episode_id)
        if not self._episode_index:
            self._load_episode_index()

        if self._df is None:
            raise RuntimeError("Internal dataframe is not loaded.")

        if episode_id is not None:
            selected_episode_id = str(episode_id)
            if selected_episode_id not in self._episode_index:
                raise KeyError(
                    f"episode_id '{selected_episode_id}' not found in loaded parquet."
                )
        else:
            # Deterministic default: first episode_id in sorted index mapping.
            selected_episode_id = next(iter(self._episode_index))

        episode_rows = self._episode_index[selected_episode_id]
        episode_df = self._df.iloc[episode_rows].reset_index(drop=True)
        if episode_df.empty:
            raise RuntimeError(
                f"Selected episode '{selected_episode_id}' has no rows after slicing."
            )

        self._current_episode_id = selected_episode_id
        self._current_episode_df = episode_df
        self._current_row_ptr = 0

        # Placeholder bookkeeping reset; mechanics are implemented in a later step.
        self._sold_fraction = 0.0
        self._remaining_fraction = 1.0
        self._cum_realized_pre_tax_pnl = 0.0
        self._cum_realized_after_tax_pnl = 0.0
        self._last_sale_date = None
        self._sale_count = 0

        observation = self._get_observation()
        current_row = self._current_episode_df.iloc[self._current_row_ptr]
        initial_tax_regime = self._classify_tax_regime(current_row)
        initial_full_position_pnl = self._get_full_position_pnl(current_row)
        (
            self._after_tax_liquidation_value_remaining,
            initial_liquidation_tax_rate,
            initial_liquidation_pre_tax_value,
            initial_liquidation_tax_drag,
        ) = self._compute_after_tax_liquidation_value(
            remaining_fraction=self._remaining_fraction,
            full_position_pnl=initial_full_position_pnl,
            tax_regime=initial_tax_regime,
        )
        self._after_tax_total_value = float(
            self._cum_realized_after_tax_pnl
            + self._after_tax_liquidation_value_remaining
        )
        self._prev_after_tax_total_value = self._after_tax_total_value

        info: InfoDict = {
            "episode_id": self._current_episode_id,
            "current_row_ptr": self._current_row_ptr,
            "date": current_row["date"],
            "tax_transition_date": current_row["tax_transition_date"],
            "sold_fraction": self._sold_fraction,
            "remaining_fraction": self._remaining_fraction,
            "cum_realized_pre_tax_pnl": self._cum_realized_pre_tax_pnl,
            "cum_realized_after_tax_pnl": self._cum_realized_after_tax_pnl,
            "tax_profile_name": self.tax_config.get(
                "profile_name", "unspecified_individual"
            ),
            "tax_regime": initial_tax_regime,
            "full_position_pnl": initial_full_position_pnl,
            "after_tax_liquidation_tax_regime": initial_tax_regime,
            "after_tax_liquidation_tax_rate": initial_liquidation_tax_rate,
            "after_tax_liquidation_pre_tax_value_remaining": (
                initial_liquidation_pre_tax_value
            ),
            "after_tax_liquidation_tax_drag_remaining": initial_liquidation_tax_drag,
            "after_tax_liquidation_value_remaining": (
                self._after_tax_liquidation_value_remaining
            ),
            "after_tax_total_value": self._after_tax_total_value,
            "previous_after_tax_total_value": self._prev_after_tax_total_value,
            "reward_version": self.reward_version,
            "reward": 0.0,
            "reward_A": 0.0,
            "reward_C_lite": 0.0,
            "reward_C_lite_v2": 0.0,
            "transaction_penalty": 0.0,
            "transaction_penalty_applied": False,
            "lambda_transaction": self._lambda_transaction,
            "cooldown_penalty": 0.0,
            "cooldown_penalty_applied": False,
            "cooldown_days": self._cooldown_days,
            "lambda_cooldown": self._lambda_cooldown,
            "days_since_last_sale": None,
            "last_sale_date_before_step": None,
            "last_sale_date_after_step": None,
            "sale_count": self._sale_count,
            "previous_sale_exists": False,
            "is_automatic_terminal_liquidation": False,
        }
        return observation, info

    def _get_observation(self) -> Observation:
        """Return the current observation from supplied frozen state columns only."""
        if self._current_episode_df is None or self._current_episode_df.empty:
            raise RuntimeError("No active episode. Call reset() before _get_observation().")

        if not (0 <= self._current_row_ptr < len(self._current_episode_df)):
            raise IndexError(
                "Current row pointer is out of range for active episode: "
                f"{self._current_row_ptr}"
            )

        row = self._current_episode_df.iloc[self._current_row_ptr]
        try:
            return row.loc[self.state_columns].to_numpy(dtype=np.float32, copy=True)
        except KeyError as exc:
            raise KeyError(
                "Active episode is missing one or more state_columns during observation extraction."
            ) from exc

    def step(self, action: int) -> StepResult:
        """Advance one step using a discrete sell-fraction action.

        Args:
            action: Index into `self.action_fractions`, where each element is the
                target sell fraction of the original position.

        Returns:
            `(observation, reward, done, truncated, info)` where:
            - reward is Reward A: change in after-tax total position value
            - done follows environment terminal conditions
            - truncated is reserved for artificial cutoffs (currently always False)
        """
        if self._brazil_mode:
            return self._step_brazil(action)
        if self._current_episode_df is None or self._current_episode_df.empty:
            raise RuntimeError("No active episode. Call reset() before step().")

        if not isinstance(action, (int, np.integer)):
            raise TypeError(
                f"Action must be an integer index, got {type(action).__name__}."
            )
        action_idx = int(action)

        if not (0 <= action_idx < len(self.action_fractions)):
            raise ValueError(
                f"Invalid action index {action_idx}. "
                f"Expected an integer in [0, {len(self.action_fractions) - 1}]."
            )

        sale_row_ptr = self._current_row_ptr
        sale_row = self._current_episode_df.iloc[sale_row_ptr]
        tax_regime = self._classify_tax_regime(sale_row)
        full_position_pnl = self._get_full_position_pnl(sale_row)

        requested_fraction = float(self.action_fractions[action_idx])
        # Actions are fractions of the original position. Execution is clamped by
        # remaining inventory to prevent overselling.
        executable_fraction = float(min(requested_fraction, self._remaining_fraction))
        executable_fraction = float(
            np.clip(executable_fraction, 0.0, self._remaining_fraction)
        )

        # Require rates when a positive realized sale (taxable gain) occurs.
        # For zero-execution (hold) paths, keep rate optional.
        realized_pre_tax_increment = float(executable_fraction * full_position_pnl)

        require_rates = bool(executable_fraction > 0.0 and realized_pre_tax_increment > 0.0)
        applicable_tax_rate = self._resolve_effective_tax_rate(
            tax_regime,
            require_rates=require_rates,
        )
        tax_paid, realized_after_tax_increment = self._compute_after_tax_pnl_increment(
            realized_pre_tax_increment,
            applicable_tax_rate,
        )

        self._cum_realized_pre_tax_pnl = float(
            self._cum_realized_pre_tax_pnl + realized_pre_tax_increment
        )
        self._cum_realized_after_tax_pnl = float(
            self._cum_realized_after_tax_pnl + realized_after_tax_increment
        )

        sold_fraction = float(self._sold_fraction + executable_fraction)
        sold_fraction = float(np.clip(sold_fraction, 0.0, 1.0))
        if np.isclose(sold_fraction, 1.0, atol=1e-12):
            sold_fraction = 1.0
        self._sold_fraction = sold_fraction

        remaining_fraction = float(1.0 - self._sold_fraction)
        if np.isclose(remaining_fraction, 0.0, atol=1e-12):
            remaining_fraction = 0.0
        self._remaining_fraction = float(max(0.0, remaining_fraction))

        last_row_ptr = len(self._current_episode_df) - 1
        next_row_ptr = (
            self._current_row_ptr + 1
            if self._current_row_ptr < last_row_ptr
            else self._current_row_ptr
        )
        done = (next_row_ptr >= last_row_ptr) or (self._remaining_fraction == 0.0)
        truncated = False

        # Terminal unsold inventory is valued with long-term treatment, then
        # converted into realized bookkeeping below without changing total value.
        terminal_liquidation_required = bool(done and self._remaining_fraction > 0.0)
        liquidation_row_ptr = (
            next_row_ptr if terminal_liquidation_required else sale_row_ptr
        )
        liquidation_row = self._current_episode_df.iloc[liquidation_row_ptr]
        liquidation_full_position_pnl = (
            self._get_full_position_pnl(liquidation_row)
            if terminal_liquidation_required
            else full_position_pnl
        )
        liquidation_tax_regime = (
            "long_term" if terminal_liquidation_required else tax_regime
        )
        (
            after_tax_liquidation_value_remaining,
            after_tax_liquidation_tax_rate,
            after_tax_liquidation_pre_tax_value_remaining,
            after_tax_liquidation_tax_drag_remaining,
        ) = self._compute_after_tax_liquidation_value(
            remaining_fraction=self._remaining_fraction,
            full_position_pnl=liquidation_full_position_pnl,
            tax_regime=liquidation_tax_regime,
        )

        previous_after_tax_total_value = self._prev_after_tax_total_value
        after_tax_total_value = float(
            self._cum_realized_after_tax_pnl
            + after_tax_liquidation_value_remaining
        )
        reward_A = float(after_tax_total_value - previous_after_tax_total_value)

        sale_executed = bool(executable_fraction > 0.0)
        current_sale_date = (
            self._parse_step_date(sale_row["date"], context="current sale row")
            if sale_executed
            else None
        )
        last_sale_date_before_step = self._last_sale_date
        (
            cooldown_penalty,
            cooldown_penalty_applied,
            days_since_last_sale,
            previous_sale_exists,
        ) = self._compute_cooldown_penalty(
            executable_fraction=executable_fraction,
            current_sale_date=current_sale_date,
        )
        (
            transaction_penalty,
            transaction_penalty_applied,
        ) = self._compute_transaction_penalty(
            executable_fraction=executable_fraction,
            is_automatic_terminal_liquidation=False,
        )
        reward_C_lite = float(reward_A - cooldown_penalty)
        reward_C_lite_v2 = float(
            reward_A - transaction_penalty - cooldown_penalty
        )
        if self.reward_version == REWARD_C_LITE_VERSION:
            reward = reward_C_lite
        elif self.reward_version == REWARD_C_LITE_V2_VERSION:
            reward = reward_C_lite_v2
        else:
            reward = reward_A

        self._after_tax_liquidation_value_remaining = (
            after_tax_liquidation_value_remaining
        )
        self._after_tax_total_value = after_tax_total_value
        self._prev_after_tax_total_value = after_tax_total_value

        final_liquidation_pre_tax_value_remaining = (
            after_tax_liquidation_pre_tax_value_remaining
        )
        final_liquidation_tax_drag_remaining = (
            after_tax_liquidation_tax_drag_remaining
        )

        terminal_liquidation_tax_rate: float | None = None
        terminal_liquidation_pre_tax_increment = 0.0
        terminal_liquidation_tax_paid = 0.0
        terminal_liquidation_after_tax_increment = 0.0

        if terminal_liquidation_required:
            terminal_liquidation_tax_rate = self._resolve_effective_tax_rate(
                "long_term",
                require_rates=bool(
                    after_tax_liquidation_pre_tax_value_remaining > 0.0
                ),
            )
            terminal_liquidation_pre_tax_increment = (
                after_tax_liquidation_pre_tax_value_remaining
            )
            (
                terminal_liquidation_tax_paid,
                terminal_liquidation_after_tax_increment,
            ) = self._compute_after_tax_pnl_increment(
                terminal_liquidation_pre_tax_increment,
                terminal_liquidation_tax_rate,
            )
            self._cum_realized_pre_tax_pnl = float(
                self._cum_realized_pre_tax_pnl
                + terminal_liquidation_pre_tax_increment
            )
            self._cum_realized_after_tax_pnl = float(
                self._cum_realized_after_tax_pnl
                + terminal_liquidation_after_tax_increment
            )
            self._sold_fraction = 1.0
            self._remaining_fraction = 0.0
            self._after_tax_liquidation_value_remaining = 0.0
            self._after_tax_total_value = self._cum_realized_after_tax_pnl
            self._prev_after_tax_total_value = self._after_tax_total_value
            final_liquidation_pre_tax_value_remaining = 0.0
            final_liquidation_tax_drag_remaining = 0.0

        if sale_executed:
            self._sale_count += 1
            self._last_sale_date = current_sale_date
        last_sale_date_after_step = self._last_sale_date

        self._current_row_ptr = next_row_ptr
        observation = self._get_observation()
        info: InfoDict = {
            "episode_id": self._current_episode_id,
            "current_row_ptr": self._current_row_ptr,
            "sale_row_ptr": sale_row_ptr,
            "date": sale_row["date"],
            "tax_transition_date": sale_row["tax_transition_date"],
            "action": action_idx,
            "action_fraction_requested": requested_fraction,
            "action_fraction_executed": executable_fraction,
            "sold_fraction": self._sold_fraction,
            "remaining_fraction": self._remaining_fraction,
            "tax_profile_name": self.tax_config.get(
                "profile_name", "unspecified_individual"
            ),
            "tax_regime": tax_regime,
            "applicable_tax_rate": applicable_tax_rate,
            "full_position_pnl": full_position_pnl,
            "realized_pre_tax_increment": realized_pre_tax_increment,
            "tax_paid": tax_paid,
            "realized_after_tax_increment": realized_after_tax_increment,
            "cum_realized_pre_tax_pnl": self._cum_realized_pre_tax_pnl,
            "cum_realized_after_tax_pnl": self._cum_realized_after_tax_pnl,
            "previous_after_tax_total_value": previous_after_tax_total_value,
            "after_tax_liquidation_tax_regime": liquidation_tax_regime,
            "after_tax_liquidation_tax_rate": after_tax_liquidation_tax_rate,
            "after_tax_liquidation_pre_tax_value_remaining": (
                final_liquidation_pre_tax_value_remaining
            ),
            "after_tax_liquidation_tax_drag_remaining": (
                final_liquidation_tax_drag_remaining
            ),
            "after_tax_liquidation_value_remaining": (
                self._after_tax_liquidation_value_remaining
            ),
            "after_tax_total_value": self._after_tax_total_value,
            "reward_version": self.reward_version,
            "reward": reward,
            "reward_A": reward_A,
            "reward_C_lite": reward_C_lite,
            "reward_C_lite_v2": reward_C_lite_v2,
            "transaction_penalty": transaction_penalty,
            "transaction_penalty_applied": transaction_penalty_applied,
            "lambda_transaction": self._lambda_transaction,
            "cooldown_penalty": cooldown_penalty,
            "cooldown_penalty_applied": cooldown_penalty_applied,
            "cooldown_days": self._cooldown_days,
            "lambda_cooldown": self._lambda_cooldown,
            "days_since_last_sale": days_since_last_sale,
            "last_sale_date_before_step": last_sale_date_before_step,
            "last_sale_date_after_step": last_sale_date_after_step,
            "sale_count": self._sale_count,
            "previous_sale_exists": previous_sale_exists,
            "is_automatic_terminal_liquidation": terminal_liquidation_required,
            "terminal_liquidation_executed": terminal_liquidation_required,
            "terminal_liquidation_row_ptr": (
                liquidation_row_ptr if terminal_liquidation_required else None
            ),
            "terminal_liquidation_date": (
                liquidation_row["date"] if terminal_liquidation_required else None
            ),
            "terminal_liquidation_full_position_pnl": (
                liquidation_full_position_pnl
                if terminal_liquidation_required
                else None
            ),
            "terminal_liquidation_tax_regime": (
                "long_term" if terminal_liquidation_required else None
            ),
            "terminal_liquidation_tax_rate": terminal_liquidation_tax_rate,
            "terminal_liquidation_pre_tax_increment": (
                terminal_liquidation_pre_tax_increment
            ),
            "terminal_liquidation_tax_paid": terminal_liquidation_tax_paid,
            "terminal_liquidation_after_tax_increment": (
                terminal_liquidation_after_tax_increment
            ),
            "done": done,
            "truncated": truncated,
        }
        return observation, reward, done, truncated, info

    def _reset_brazil(self, episode_id: str | None) -> ResetResult:
        if not self._episode_index:
            self._load_episode_index()
        if self._df is None:
            raise RuntimeError("Internal dataframe is not loaded.")
        selected_id = str(episode_id) if episode_id is not None else next(iter(self._episode_index))
        if selected_id not in self._episode_index:
            raise KeyError(f"episode_id '{selected_id}' not found in loaded parquet.")
        episode = self._df.iloc[self._episode_index[selected_id]].reset_index(drop=True)
        self._current_episode_id = selected_id
        self._current_episode_df = episode
        self._current_row_ptr = 0
        self._remaining_fraction = 1.0
        self._sold_fraction = 0.0
        self._sale_count = 0
        self._last_sale_date = None
        self._brazil_first_sale_date = None
        self._brazil_mandatory_sale_count = 0
        self._brazil_lots = []
        self._brazil_done = False
        self._brazil_previous_observation_date = pd.Timestamp(episode.iloc[0]["date"])
        self._brazil_cash_interest_cumulative = 0.0
        self._brazil_fixed_income_tax_cumulative = 0.0
        self._brazil_equity_tax_cumulative = 0.0
        self._brazil_gross_proceeds_cumulative = 0.0
        self._brazil_cash_principal_cumulative = 0.0

        row = episode.iloc[0]
        price_ratio = float(row["adj_close"] / row["simulated_purchase_price"])
        initial_wealth = calculate_sale(1.0, price_ratio).cash_deposit
        self._brazil_initial_wealth = initial_wealth
        self._brazil_previous_wealth = initial_wealth
        info: InfoDict = {
            "episode_id": selected_id,
            "date": row["date"],
            "terminal_valuation_date": None,
            "current_row_ptr": 0,
            "scenario_name": "brazil_inspired_v1",
            "reward_version": self.reward_version,
            "base_reward_version": self.base_reward_version,
            "initial_total_after_tax_wealth": initial_wealth,
            "previous_total_after_tax_wealth": initial_wealth,
            "total_after_tax_wealth": initial_wealth,
            "base_reward": 0.0,
            "reward": 0.0,
            "shaped_reward": 0.0,
            "cash_balance": 0.0,
            "cash_balance_gross": 0.0,
            "cash_principal_cumulative": 0.0,
            "cash_interest_gross_step": 0.0,
            "cash_interest_gross_cumulative": 0.0,
            "fixed_income_tax_estimated_liability": 0.0,
            "fixed_income_tax_step": 0.0,
            "fixed_income_tax_cumulative": 0.0,
            "gross_sale_proceeds": 0.0,
            "gross_sale_proceeds_cumulative": 0.0,
            "equity_tax_step": 0.0,
            "equity_tax_cumulative": 0.0,
            "remaining_fraction": 1.0,
            "remaining_inventory_value": price_ratio,
            "after_tax_liquidation_value": initial_wealth,
            "discretionary_sale_count": 0,
            "mandatory_sale_count": 0,
            "first_sale_date": None,
            "first_sale_days_from_start": None,
            "cash_lot_count": 0,
            "done": False,
            "truncated": False,
        }
        return self._get_observation(), info

    def _step_brazil(self, action: int) -> StepResult:
        if self._current_episode_df is None or self._current_episode_df.empty:
            raise RuntimeError("No active episode. Call reset() before step().")
        if self._brazil_done:
            raise RuntimeError("Brazil episode is complete. Call reset() before step().")
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise TypeError(f"Action must be an integer index, got {type(action).__name__}.")
        action_idx = int(action)
        if not 0 <= action_idx < len(self.action_fractions):
            raise ValueError(f"Invalid action index {action_idx}.")

        row_ptr = self._current_row_ptr
        row = self._current_episode_df.iloc[row_ptr]
        current_date = pd.Timestamp(row["date"])
        price_ratio = float(row["adj_close"] / row["simulated_purchase_price"])
        rate = float(self.economic_scenario["cash_account"]["annual_gross_rate"])
        last_row_ptr = len(self._current_episode_df) - 1
        final_row = row_ptr == last_row_ptr
        previous_wealth = self._brazil_previous_wealth
        if self._brazil_previous_observation_date is not None and current_date < self._brazil_previous_observation_date:
            raise RuntimeError("Brazil observations must advance chronologically.")

        interest_step = 0.0
        if row_ptr > 0:
            for lot in self._brazil_lots:
                before = gross_lot_value(lot.principal, rate, lot.market_day_age)
                lot.market_day_age += 1
                after = gross_lot_value(lot.principal, rate, lot.market_day_age)
                interest_step += after - before
        requested_fraction = float(self.action_fractions[action_idx])
        executed_fraction = float(np.clip(
            min(requested_fraction, self._remaining_fraction), 0.0, self._remaining_fraction
        ))
        discretionary = calculate_sale(executed_fraction, price_ratio)
        if executed_fraction > 0.0:
            self._brazil_lots.append(_BrazilCashLot(discretionary.cash_deposit, current_date))
            self._brazil_cash_principal_cumulative += discretionary.cash_deposit
        self._remaining_fraction = float(max(0.0, self._remaining_fraction - executed_fraction))
        if np.isclose(self._remaining_fraction, 0.0, atol=1e-12):
            self._remaining_fraction = 0.0
        self._sold_fraction = 1.0 - self._remaining_fraction

        last_sale_date_before_step = self._last_sale_date
        cooldown_penalty, cooldown_applied, days_since_last_sale, previous_sale_exists = (
            self._compute_cooldown_penalty(
                executable_fraction=executed_fraction,
                current_sale_date=current_date if executed_fraction > 0.0 else None,
            )
        )
        if executed_fraction > 0.0:
            if self._sale_count == 0:
                self._brazil_first_sale_date = current_date
            self._sale_count += 1
            self._last_sale_date = current_date

        early_full_sale = self._remaining_fraction == 0.0 and not final_row
        done = final_row or early_full_sale
        terminal_valuation_date = (
            pd.Timestamp(self._current_episode_df.iloc[last_row_ptr]["date"]) if done else None
        )
        if early_full_sale:
            # Settle at the fixed horizon without exposing later rows to the policy.
            remaining_intervals = last_row_ptr - row_ptr
            for lot in self._brazil_lots:
                before = gross_lot_value(lot.principal, rate, lot.market_day_age)
                lot.market_day_age += remaining_intervals
                after = gross_lot_value(lot.principal, rate, lot.market_day_age)
                interest_step += after - before
        self._brazil_cash_interest_cumulative += interest_step

        gross_cash = 0.0
        after_tax_cash = 0.0
        estimated_interest_tax = 0.0
        valuation_date = terminal_valuation_date if done else current_date
        for lot in self._brazil_lots:
            days_held = int((valuation_date - lot.deposit_date).days)
            value = after_tax_lot_value(lot.principal, rate, lot.market_day_age, days_held)
            gross_cash += value.gross_value
            after_tax_cash += value.after_tax_value
            estimated_interest_tax += value.interest_tax

        remaining_fraction_before_terminal = self._remaining_fraction
        inventory_gross_before_terminal = remaining_fraction_before_terminal * price_ratio
        inventory_net_before_terminal = calculate_sale(
            remaining_fraction_before_terminal, price_ratio
        ).cash_deposit
        mandatory = calculate_sale(0.0, price_ratio)
        fixed_income_tax_step = 0.0
        if done:
            fixed_income_tax_step = estimated_interest_tax
            self._brazil_fixed_income_tax_cumulative += fixed_income_tax_step
            if final_row:
                mandatory = calculate_sale(remaining_fraction_before_terminal, price_ratio)
                if remaining_fraction_before_terminal > 0.0:
                    self._brazil_mandatory_sale_count += 1
                after_tax_cash += mandatory.cash_deposit
                gross_cash += mandatory.cash_deposit
            self._brazil_lots.clear()
            self._remaining_fraction = 0.0
            self._sold_fraction = 1.0
            inventory_gross = 0.0
            inventory_net = 0.0
            estimated_interest_tax = 0.0
        else:
            inventory_gross = inventory_gross_before_terminal
            inventory_net = inventory_net_before_terminal

        gross_proceeds_step = discretionary.gross_proceeds + mandatory.gross_proceeds
        equity_tax_step = discretionary.equity_tax + mandatory.equity_tax
        self._brazil_gross_proceeds_cumulative += gross_proceeds_step
        self._brazil_equity_tax_cumulative += equity_tax_step
        wealth = after_tax_cash + inventory_net
        base_reward = wealth - previous_wealth
        reward = base_reward - cooldown_penalty if self.reward_version == BRAZIL_C_LITE_REWARD_VERSION else base_reward
        self._brazil_previous_wealth = wealth
        self._brazil_previous_observation_date = current_date

        first_sale_days = (
            int((self._brazil_first_sale_date - pd.Timestamp(self._current_episode_df.iloc[0]["date"])).days)
            if self._brazil_first_sale_date is not None else None
        )
        info: InfoDict = {
            "episode_id": self._current_episode_id,
            "date": current_date,
            "terminal_valuation_date": terminal_valuation_date,
            "current_row_ptr": row_ptr,
            "scenario_name": "brazil_inspired_v1",
            "reward_version": self.reward_version,
            "base_reward_version": self.base_reward_version,
            "action": action_idx,
            "action_fraction_requested": requested_fraction,
            "action_fraction_executed": executed_fraction,
            "discretionary_sale_fraction_executed": executed_fraction,
            "mandatory_sale_fraction_executed": remaining_fraction_before_terminal if final_row else 0.0,
            "sold_fraction": self._sold_fraction,
            "remaining_fraction": self._remaining_fraction,
            "remaining_fraction_before_terminal": remaining_fraction_before_terminal if done else None,
            "discretionary_sale_count": self._sale_count,
            "mandatory_sale_count": self._brazil_mandatory_sale_count,
            "sale_count": self._sale_count,
            "first_sale_date": self._brazil_first_sale_date,
            "first_sale_days_from_start": first_sale_days,
            "last_sale_date_before_step": last_sale_date_before_step,
            "last_sale_date_after_step": self._last_sale_date,
            "days_since_last_sale": days_since_last_sale,
            "previous_sale_exists": previous_sale_exists,
            "cooldown_penalty": cooldown_penalty,
            "cooldown_penalty_applied": cooldown_applied,
            "initial_total_after_tax_wealth": self._brazil_initial_wealth,
            "previous_total_after_tax_wealth": previous_wealth,
            "total_after_tax_wealth": wealth,
            "base_reward": base_reward,
            "shaped_reward": reward,
            "reward": reward,
            "cash_balance": after_tax_cash,
            "cash_balance_gross": gross_cash,
            "cash_principal_cumulative": self._brazil_cash_principal_cumulative,
            "cash_interest_gross_step": interest_step,
            "cash_interest_gross_cumulative": self._brazil_cash_interest_cumulative,
            "fixed_income_tax_estimated_liability": estimated_interest_tax,
            "fixed_income_tax_step": fixed_income_tax_step,
            "fixed_income_tax_cumulative": self._brazil_fixed_income_tax_cumulative,
            "cash_lot_count": len(self._brazil_lots),
            "gross_sale_proceeds": gross_proceeds_step,
            "gross_sale_proceeds_cumulative": self._brazil_gross_proceeds_cumulative,
            "discretionary_gross_sale_proceeds": discretionary.gross_proceeds,
            "mandatory_gross_sale_proceeds": mandatory.gross_proceeds,
            "equity_tax_step": equity_tax_step,
            "equity_tax_cumulative": self._brazil_equity_tax_cumulative,
            "discretionary_equity_tax": discretionary.equity_tax,
            "mandatory_equity_tax": mandatory.equity_tax,
            "discretionary_cash_deposit": discretionary.cash_deposit,
            "mandatory_after_tax_proceeds": mandatory.cash_deposit,
            "remaining_inventory_value": inventory_gross,
            "after_tax_liquidation_value": inventory_net,
            "remaining_inventory_value_before_terminal": inventory_gross_before_terminal if done else None,
            "after_tax_liquidation_value_before_terminal": inventory_net_before_terminal if done else None,
            "is_automatic_terminal_liquidation": bool(final_row and remaining_fraction_before_terminal > 0.0),
            "done": done,
            "truncated": False,
        }
        self._brazil_done = done
        if not done:
            self._current_row_ptr += 1
        return self._get_observation(), reward, done, False, info
