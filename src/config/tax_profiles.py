"""Shared tax-profile config loading helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


TAX_PROFILE_FIELDS = (
    "profile_name",
    "short_term_rate",
    "long_term_rate",
    "niit_rate",
    "apply_niit",
)

PROFILE_PATH_KEYS = (
    "config_path",
    "profiles_path",
    "profile_config_path",
    "tax_profile_config_path",
)

TOP_LEVEL_PROFILE_PATH_KEYS = (
    "tax_profile_config_path",
    "tax_profiles_path",
)

BRAZIL_V1_CASH_RATES = frozenset({0.07, 0.10, 0.105, 0.12, 0.1375, 0.15})


def _resolve_path(path_value: str | Path, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _parse_scalar(value: str) -> Any:
    normalized = value.strip()
    if normalized in {"true", "True"}:
        return True
    if normalized in {"false", "False"}:
        return False
    if (
        (normalized.startswith('"') and normalized.endswith('"'))
        or (normalized.startswith("'") and normalized.endswith("'"))
    ):
        return normalized[1:-1]
    try:
        return float(normalized)
    except ValueError:
        return normalized


def _parse_key_value(text: str) -> tuple[str, Any]:
    if ":" not in text:
        raise ValueError(f"Invalid tax profile YAML line: {text!r}")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid tax profile YAML key in line: {text!r}")
    return key, _parse_scalar(value)


def _load_simple_profile_yaml(text: str) -> Mapping[str, Any]:
    profiles: list[dict[str, Any]] = []
    current_profile: dict[str, Any] | None = None
    seen_profiles_key = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped == "profiles:":
            seen_profiles_key = True
            continue

        if stripped.startswith("- "):
            if not seen_profiles_key:
                raise ValueError("Tax profile YAML list must be under 'profiles:'.")
            if current_profile is not None:
                profiles.append(current_profile)
            current_profile = {}
            remainder = stripped[2:].strip()
            if remainder:
                key, value = _parse_key_value(remainder)
                current_profile[key] = value
            continue

        if current_profile is None:
            raise ValueError(
                "Tax profile YAML must contain list items under 'profiles:'."
            )
        key, value = _parse_key_value(stripped)
        current_profile[key] = value

    if current_profile is not None:
        profiles.append(current_profile)
    if not seen_profiles_key:
        raise ValueError("Tax profile YAML must define a top-level 'profiles:' key.")
    return {"profiles": profiles}


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:  # pragma: no cover - dependency guard.
                payload = _load_simple_profile_yaml(handle.read())
            else:
                payload = yaml.safe_load(handle)
        else:
            raise ValueError(
                f"Unsupported tax profile config extension {path.suffix!r}; "
                "expected .json, .yaml, or .yml."
            )

    if not isinstance(payload, Mapping):
        raise ValueError(f"Tax profile config must be a mapping: {path}")
    return payload


def _coerce_rate(profile: Mapping[str, Any], key: str) -> float:
    try:
        value = float(profile[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Tax profile {profile.get('profile_name', '<unknown>')!r} "
            f"must define numeric {key!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Tax profile {profile.get('profile_name', '<unknown>')!r} "
            f"has non-finite {key!r}: {value!r}."
        )
    if value < 0.0:
        raise ValueError(
            f"Tax profile {profile.get('profile_name', '<unknown>')!r} "
            f"has negative {key!r}: {value!r}."
        )
    return value


def _coerce_bool(profile: Mapping[str, Any], key: str) -> bool:
    try:
        value = profile[key]
    except KeyError as exc:
        raise ValueError(
            f"Tax profile {profile.get('profile_name', '<unknown>')!r} "
            f"must define {key!r}."
        ) from exc
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(
        f"Tax profile {profile.get('profile_name', '<unknown>')!r} "
        f"must define boolean {key!r}."
    )


def validate_tax_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in TAX_PROFILE_FIELDS if field not in profile]
    if missing_fields:
        raise ValueError(
            f"Tax profile is missing required field(s): {missing_fields}."
        )

    profile_name = profile["profile_name"]
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("Tax profile field 'profile_name' must be a non-empty string.")

    return {
        "profile_name": profile_name,
        "short_term_rate": _coerce_rate(profile, "short_term_rate"),
        "long_term_rate": _coerce_rate(profile, "long_term_rate"),
        "niit_rate": _coerce_rate(profile, "niit_rate"),
        "apply_niit": _coerce_bool(profile, "apply_niit"),
    }


def load_tax_profiles(
    path: str | Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    resolved_path = _resolve_path(path, base_dir)
    payload = _load_mapping(resolved_path)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(
            f"Tax profile config must contain a non-empty 'profiles' list: "
            f"{resolved_path}"
        )

    profiles: dict[str, dict[str, Any]] = {}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, Mapping):
            raise ValueError(
                f"Each tax profile must be a mapping in config: {resolved_path}"
            )
        profile = validate_tax_profile(raw_profile)
        profile_name = str(profile["profile_name"])
        if profile_name in profiles:
            raise ValueError(
                f"Duplicate tax profile {profile_name!r} in config: {resolved_path}"
            )
        profiles[profile_name] = profile

    return profiles


def load_tax_profile(
    path: str | Path,
    profile_name: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    profiles = load_tax_profiles(path, base_dir=base_dir)
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Tax profile {profile_name!r} not found in {path}. "
            f"Available profiles: {available}"
        )
    return dict(profiles[profile_name])


def resolve_tax_profile_from_config(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    tax_profile_config = config.get("tax_profile")

    if isinstance(tax_profile_config, Mapping):
        path_value = next(
            (
                tax_profile_config[key]
                for key in PROFILE_PATH_KEYS
                if tax_profile_config.get(key) is not None
            ),
            None,
        )
        profile_name = tax_profile_config.get("profile_name")
        if path_value is not None:
            if not isinstance(profile_name, str) or not profile_name:
                raise ValueError(
                    "tax_profile.profile_name is required when tax_profile "
                    "references a profile config file."
                )
            return load_tax_profile(path_value, profile_name, base_dir=base_dir)

        if all(field in tax_profile_config for field in TAX_PROFILE_FIELDS):
            return validate_tax_profile(tax_profile_config)

    top_level_path = next(
        (
            config[key]
            for key in TOP_LEVEL_PROFILE_PATH_KEYS
            if config.get(key) is not None
        ),
        None,
    )
    top_level_name = config.get("tax_profile_name")
    if isinstance(tax_profile_config, str):
        top_level_name = tax_profile_config

    if top_level_path is not None:
        if not isinstance(top_level_name, str) or not top_level_name:
            raise ValueError(
                "tax_profile_name is required when using a top-level tax "
                "profile config path."
            )
        return load_tax_profile(top_level_path, top_level_name, base_dir=base_dir)

    raise ValueError(
        "Tax profile config must either define inline rates under tax_profile "
        "or reference a profile file with tax_profile.config_path and "
        "tax_profile.profile_name."
    )


def _scenario_mapping(parent: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _scenario_exact(parent: Mapping[str, Any], key: str, expected: Any, path: str) -> Any:
    if key not in parent:
        raise ValueError(f"{path} is required.")
    value = parent[key]
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{path} must be {expected!r}; got {value!r}.")
    return value


def _scenario_rate(parent: Mapping[str, Any], key: str, path: str) -> float:
    if key not in parent:
        raise ValueError(f"{path} is required.")
    value = parent[key]
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite numeric rate.")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite numeric rate.") from exc
    if not math.isfinite(rate):
        raise ValueError(f"{path} must be a finite numeric rate.")
    return rate


def _scenario_allowed_keys(parent: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unexpected = sorted(set(parent) - allowed)
    if unexpected:
        raise ValueError(f"{path} has unsupported field(s): {unexpected}.")


def resolve_economic_scenario_from_config(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate and normalize an explicit economic scenario, if configured.

    This resolver is independent of the legacy U.S. tax-profile loader. Later
    phases will pass its result to training, evaluation, and the environment.
    """
    if "economic_scenario" not in config:
        return None

    scenario = _scenario_mapping(config, "economic_scenario", "economic_scenario")
    _scenario_allowed_keys(
        scenario,
        {"name", "equity_tax", "cash_account", "terminal_horizon", "currency", "fx_conversion"},
        "economic_scenario",
    )
    name = _scenario_exact(scenario, "name", "brazil_inspired_v1", "economic_scenario.name")

    legacy_fields = (
        "tax_profile", "tax_profile_name", "tax_profile_config_path",
        "tax_profiles_path", "short_term_rate", "long_term_rate",
    )
    for key in legacy_fields:
        if key in config:
            raise ValueError(
                f"{key} contradicts economic_scenario.name={name!r}; "
                "remove legacy U.S. tax-profile fields from the Brazil config."
            )

    equity = _scenario_mapping(scenario, "equity_tax", "economic_scenario.equity_tax")
    for key in ("short_term_rate", "long_term_rate"):
        if key in equity:
            raise ValueError(f"economic_scenario.equity_tax.{key} contradicts flat_positive_gains.")
    _scenario_allowed_keys(equity, {"regime", "rate", "loss_credit"}, "economic_scenario.equity_tax")
    _scenario_exact(equity, "regime", "flat_positive_gains", "economic_scenario.equity_tax.regime")
    equity_rate = _scenario_rate(equity, "rate", "economic_scenario.equity_tax.rate")
    if not 0.0 <= equity_rate <= 1.0 or equity_rate != 0.15:
        raise ValueError("economic_scenario.equity_tax.rate must be 0.15 for brazil_inspired_v1 (and within [0, 1]).")
    _scenario_exact(equity, "loss_credit", False, "economic_scenario.equity_tax.loss_credit")

    cash = _scenario_mapping(scenario, "cash_account", "economic_scenario.cash_account")
    _scenario_allowed_keys(
        cash,
        {"enabled", "annual_gross_rate", "compounding", "market_days_per_year", "interest_tax"},
        "economic_scenario.cash_account",
    )
    _scenario_exact(cash, "enabled", True, "economic_scenario.cash_account.enabled")
    cash_rate = _scenario_rate(cash, "annual_gross_rate", "economic_scenario.cash_account.annual_gross_rate")
    if cash_rate not in BRAZIL_V1_CASH_RATES:
        raise ValueError(
            "economic_scenario.cash_account.annual_gross_rate must be one of "
            f"{sorted(BRAZIL_V1_CASH_RATES)} for brazil_inspired_v1."
        )
    _scenario_exact(cash, "compounding", "effective_annual_market_252", "economic_scenario.cash_account.compounding")
    _scenario_exact(cash, "market_days_per_year", 252, "economic_scenario.cash_account.market_days_per_year")
    interest_tax = _scenario_mapping(cash, "interest_tax", "economic_scenario.cash_account.interest_tax")
    _scenario_allowed_keys(interest_tax, {"regime", "tiers"}, "economic_scenario.cash_account.interest_tax")
    _scenario_exact(interest_tax, "regime", "holding_period_tiers", "economic_scenario.cash_account.interest_tax.regime")
    tiers = interest_tax.get("tiers")
    if not isinstance(tiers, list) or len(tiers) != 2:
        raise ValueError("economic_scenario.cash_account.interest_tax.tiers must contain exactly two ordered tiers.")
    for index, (days, rate) in enumerate(((180, 0.225), (None, 0.20))):
        path = f"economic_scenario.cash_account.interest_tax.tiers[{index}]"
        tier = tiers[index]
        if not isinstance(tier, Mapping):
            raise ValueError(f"{path} must be a mapping.")
        _scenario_allowed_keys(tier, {"max_calendar_days", "rate"}, path)
        _scenario_exact(tier, "max_calendar_days", days, f"{path}.max_calendar_days")
        tier_rate = _scenario_rate(tier, "rate", f"{path}.rate")
        if tier_rate != rate:
            raise ValueError(f"{path}.rate must be {rate!r}; got {tier_rate!r}.")

    _scenario_exact(scenario, "terminal_horizon", "episode_end", "economic_scenario.terminal_horizon")
    _scenario_exact(scenario, "currency", "USD", "economic_scenario.currency")
    _scenario_exact(scenario, "fx_conversion", False, "economic_scenario.fx_conversion")

    return {
        "name": name,
        "equity_tax": {"regime": "flat_positive_gains", "rate": equity_rate, "loss_credit": False},
        "cash_account": {
            "enabled": True,
            "annual_gross_rate": cash_rate,
            "compounding": "effective_annual_market_252",
            "market_days_per_year": 252,
            "interest_tax": {
                "regime": "holding_period_tiers",
                "tiers": [
                    {"max_calendar_days": 180, "rate": 0.225},
                    {"max_calendar_days": None, "rate": 0.20},
                ],
            },
        },
        "terminal_horizon": "episode_end",
        "currency": "USD",
        "fx_conversion": False,
    }
