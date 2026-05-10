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
