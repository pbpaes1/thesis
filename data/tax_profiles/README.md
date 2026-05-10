# Tax Profiles (v1)

These are thesis simulation tax profiles for the tax-aware liquidation environment.

- Profiles are reusable across the same episode-level parquet dataset.
- Profiles represent individual taxable investor scenarios only.
- This configuration is not a full legal/tax engine.
- `configs/individual_tax_profiles_v1.yaml` is the runtime config consumed by scripts.
- `individual_tax_profiles_v1.json` mirrors the same profile values as a data artifact.

Environment effective-rate rules used in the simulator:

- Short-term effective rate: `short_term_rate`
- Long-term effective rate: `long_term_rate + niit_rate` when `apply_niit=true`

## Profile Summary

| Profile | Short-Term Base | Long-Term Base | NIIT | apply_niit | Effective Short-Term | Effective Long-Term |
|---|---:|---:|---:|:---:|---:|---:|
| `tax_free` | 0.0000 | 0.0000 | 0.0000 | false | 0.0000 | 0.0000 |
| `low_income_individual` | 0.1121 | 0.0000 | 0.0000 | false | 0.1121 | 0.0000 |
| `mass_affluent_individual` | 0.1919 | 0.1018 | 0.0000 | false | 0.1919 | 0.1018 |
| `high_income_individual` | 0.2804 | 0.1335 | 0.0380 | true | 0.2804 | 0.1715 |
| `top_bracket_individual` | 0.3260 | 0.1653 | 0.0380 | true | 0.3260 | 0.2033 |
