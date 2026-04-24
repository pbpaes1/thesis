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
  and cumulative realized after-tax PnL.
- Episode end (`done`): final row reached or 100% liquidated.
- This skeleton remains generic and does not load the frozen JSON artifact itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


Observation = npt.NDArray[np.float32]
InfoDict = dict[str, Any]
ResetResult = tuple[Observation, InfoDict]
StepResult = tuple[Observation, float, bool, bool, InfoDict]


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
        seed: int | None = None,
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
                    "Taxable sale requires both 'short_term_rate' and 'long_term_rate' "
                    "in tax_config."
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

        missing_bookkeeping_cols = [
            col for col in self.REQUIRED_BOOKKEEPING_COLUMNS if col not in df.columns
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

        observation = self._get_observation()
        current_row = self._current_episode_df.iloc[self._current_row_ptr]
        info: InfoDict = {
            "episode_id": self._current_episode_id,
            "current_row_ptr": self._current_row_ptr,
            "date": current_row["date"],
            "sold_fraction": self._sold_fraction,
            "remaining_fraction": self._remaining_fraction,
            "cum_realized_pre_tax_pnl": self._cum_realized_pre_tax_pnl,
            "cum_realized_after_tax_pnl": self._cum_realized_after_tax_pnl,
            "tax_profile_name": self.tax_config.get(
                "profile_name", "unspecified_individual"
            ),
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
            - reward is currently a placeholder (0.0)
            - done follows environment terminal conditions
            - truncated is reserved for artificial cutoffs (currently always False)
        """
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

        requested_fraction = float(self.action_fractions[action_idx])
        # Actions are fractions of the original position. Execution is clamped by
        # remaining inventory to prevent overselling.
        executable_fraction = float(min(requested_fraction, self._remaining_fraction))
        executable_fraction = float(
            np.clip(executable_fraction, 0.0, self._remaining_fraction)
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

        sale_date = pd.to_datetime(sale_row["date"], errors="coerce")
        tax_transition_date = pd.to_datetime(
            sale_row["tax_transition_date"], errors="coerce"
        )
        if pd.isna(sale_date) or pd.isna(tax_transition_date):
            raise ValueError(
                "Cannot classify tax regime due to invalid 'date' or "
                "'tax_transition_date' in sale row."
            )
        tax_regime = "short_term" if sale_date < tax_transition_date else "long_term"

        # Require rates when a positive realized sale (taxable gain) occurs.
        # For zero-execution (hold) paths, keep rate optional.
        full_position_pnl = float(sale_row["unrealized_gains_pct"])
        if np.isnan(full_position_pnl):
            raise ValueError(
                "Column 'unrealized_gains_pct' contains NaN in current sale row."
            )

        realized_pre_tax_increment = float(executable_fraction * full_position_pnl)

        require_rates = bool(executable_fraction > 0.0 and realized_pre_tax_increment > 0.0)
        applicable_tax_rate = self._resolve_effective_tax_rate(
            tax_regime,
            require_rates=require_rates,
        )

        if realized_pre_tax_increment > 0.0:
            if np.isnan(applicable_tax_rate):
                raise ValueError(
                    "Taxable realized gain requires a valid applicable tax rate."
                )
            tax_paid = float(realized_pre_tax_increment * applicable_tax_rate)
        else:
            tax_paid = 0.0

        realized_after_tax_increment = float(realized_pre_tax_increment - tax_paid)

        self._cum_realized_pre_tax_pnl = float(
            self._cum_realized_pre_tax_pnl + realized_pre_tax_increment
        )
        self._cum_realized_after_tax_pnl = float(
            self._cum_realized_after_tax_pnl + realized_after_tax_increment
        )

        last_row_ptr = len(self._current_episode_df) - 1
        if self._current_row_ptr < last_row_ptr:
            self._current_row_ptr += 1

        # Step 4 terminal placeholder: terminal row reached OR fully liquidated.
        done = (self._current_row_ptr >= last_row_ptr) or (
            self._remaining_fraction == 0.0
        )
        truncated = False

        # TODO(next): replace with real reward design.
        reward = 0.0
        observation = self._get_observation()
        info: InfoDict = {
            "episode_id": self._current_episode_id,
            "current_row_ptr": self._current_row_ptr,
            "sale_row_ptr": sale_row_ptr,
            "date": sale_row["date"],
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
            "done": done,
            "truncated": truncated,
        }
        return observation, reward, done, truncated, info
