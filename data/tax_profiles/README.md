# Tax Profiles (v1)

These are thesis simulation tax profiles for the tax-aware liquidation environment.

- Profiles are reusable across the same episode-level parquet dataset.
- Profiles represent individual taxable investor scenarios only.
- This configuration is not a full legal/tax engine.

Environment effective-rate rules used in the simulator:

- Short-term effective rate: `short_term_rate`
- Long-term effective rate: `long_term_rate + niit_rate` when `apply_niit=true`

## Profile Summary

| Profile | Short-Term Base | Long-Term Base | NIIT | apply_niit | Effective Short-Term | Effective Long-Term |
|---|---:|---:|---:|:---:|---:|---:|
| `tax_free` | 0.00 | 0.00 | 0.000 | false | 0.000 | 0.000 |
| `low_income_individual` | 0.12 | 0.00 | 0.000 | false | 0.120 | 0.000 |
| `mass_affluent_individual` | 0.24 | 0.15 | 0.000 | false | 0.240 | 0.150 |
| `high_income_individual` | 0.35 | 0.15 | 0.038 | true | 0.350 | 0.188 |
| `top_bracket_individual` | 0.37 | 0.20 | 0.038 | true | 0.370 | 0.238 |
