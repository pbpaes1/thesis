# State Freeze v1 Summary

## Dataset overview

- Source file: `data\episodes\drl_episodes.parquet`
- Rows: **973,658**
- Columns: **57**

Inspection snapshot (name, dtype, examples):

| column_name | dtype | example_values |
|---|---|---|
| date | datetime64[ms] | 2010-12-31 00:00:00 | 2011-01-03 00:00:00 | 2011-01-04 00:00:00 |
| ticker | str | A | AA | AAL |
| open | float64 | 29.63519287109375 | 29.72818374633789 | 30.035764694213867 |
| high | float64 | 29.828325271606445 | 30.143062591552734 | 30.11444854736328 |
| low | float64 | 29.499284744262695 | 29.620887756347656 | 29.45636558532715 |
| close | float64 | 29.63519287109375 | 29.957082748413086 | 29.678112030029297 |
| adj_close | float64 | 26.28249168395996 | 26.567960739135742 | 26.3205509185791 |
| volume | float64 | -1.509472422013953 | 0.03252125164948387 | 0.0429876563317191 |
| source | str | sp500 | sp400 | adr |
| sma_5 | float64 | 2.787574976609339 | 2.7317692074584516 | 2.679725403837185 |
| ema_5 | float64 | 2.7459840290125066 | 2.736733343095619 | 2.6761709264755114 |
| sma_10 | float64 | 2.8401393554624765 | 2.824981065712741 | 2.7955128132569422 |
| ema_10 | float64 | 2.7788504728266927 | 2.7759629368369394 | 2.742106136289162 |
| sma_20 | float64 | 2.6589215158531307 | 2.6854520012139713 | 2.70905166677916 |
| ema_20 | float64 | 2.749113008869496 | 2.761982635497563 | 2.7548553081927625 |
| sma_50 | float64 | 2.461851630221689 | 2.48729926836551 | 2.507678414065431 |
| ema_50 | float64 | 2.7467036009988837 | 2.774926557204182 | 2.7908957037178714 |
| rsi_14 | float64 | 1.1503322986231805 | 1.326137632397428 | 0.9248837066296367 |
| macd | float64 | 1.71369917155038 | 1.6776103730160445 | 1.5799441334385835 |
| macd_signal | float64 | 0.28780402430925484 | 0.14445369811619851 | -0.16856377003674364 |
| macd_hist | float64 | 1.7503914421997924 | 1.7502418543033378 | 1.7291722239613148 |
| bb_lower | float64 | 2.2214603767146666 | 2.2807305458645826 | 2.3942035017317655 |
| bb_middle | float64 | 2.6589215158531307 | 2.6854520012139713 | 2.70905166677916 |
| bb_upper | float64 | 2.8739378904127713 | 2.8719516248873673 | 2.816745591756664 |
| Gold_Close | float64 | 1.218422313809566 | -0.6604660343011156 | -2.631054747057084 |
| WTI_Close | float64 | 1.034762807841808 | 0.08371259544824133 | -1.5356969865569217 |
| Semiconductor_Close | float64 | -0.11434572463344217 | 0.42694931641178807 | 0.01923697241608645 |
| EUR_USD_Close | float64 | 1.0039294087346764 | -0.25855387326419454 | -0.5102000594425022 |
| USD_CNY_Close | float64 | -1.5918582721595955 | 0.018348318246158724 | 2.5850966972635545 |
| USD_JPN_Close | float64 | -0.699620805514932 | 1.1834254885547009 | 0.6965852360951582 |
| Wheat_Close | float64 | 0.3266906086968619 | 0.5704987712842314 | -0.9524956249084443 |
| SPY_Close | float64 | -0.06934418154925351 | 1.0105866955018863 | -0.181588490866633 |
| VIX_Close | float64 | -0.9121624145589307 | -0.9340999078305049 | -0.9701281136510511 |
| PC1 | float64 | -6.136369575394854 | 17.25053086070744 | -13.815928522957497 |
| PC2 | float64 | 2.236080935901221 | -3.493948630617881 | 6.411597471246421 |
| PC3 | float64 | -0.8891153744832279 | 2.5281122588587417 | -1.3622629636645491 |
| PC4 | float64 | 4.198694276727404 | -0.41760680978049325 | 2.3611734356071663 |
| PC5 | float64 | -3.0220604042004253 | -0.7145641416138653 | -5.403409358482685 |
| PC6 | float64 | -1.050550024148957 | 1.6780895118413799 | -0.6343240335135363 |
| PC7 | float64 | -0.7424640540620707 | -0.8020437726960367 | 1.688154348201578 |
| PC8 | float64 | -0.39606249701491464 | -0.173475845617565 | 1.365801969300386 |
| PC9 | float64 | 0.024993735186890884 | 1.7347263011818856 | 1.3918723510306348 |
| PC10 | float64 | -1.3115927279942907 | -1.5972053799457988 | -1.900638741242179 |
| ticker_row | int64 | 251 | 252 | 253 |
| simulated_purchase_price | float64 | 17.10931396484375 | 18.650859832763672 | 22.696571350097656 |
| simulated_purchase_pos | Int64 | 166 | 167 | 171 |
| simulated_purchase_date | datetime64[ms] | 2010-08-31 00:00:00 | 2011-10-03 00:00:00 | 2012-10-23 00:00:00 |
| trigger_candidate | bool | True | False |
| valid_trigger | bool | True | False |
| episode_id | str | A_2010-08-31 | A_2011-10-03 | A_2012-10-23 |
| trigger_date | datetime64[ms] | 2010-12-31 00:00:00 | 2012-01-09 00:00:00 | 2013-05-20 00:00:00 |
| tax_transition_date | datetime64[ms] | 2011-08-31 00:00:00 | 2012-10-02 00:00:00 | 2013-10-23 00:00:00 |
| holding_period_days | int64 | 122 | 125 | 126 |
| days_until_tax_transition | int64 | 243 | 240 | 239 |
| unrealized_gains_pct | float64 | 0.5361511126609326 | 0.5528361215258331 | 0.5383755872773519 |
| days_to_tax_transition_norm | float64 | 0.6657534246575343 | 0.6575342465753424 | 0.6547945205479452 |
| unrealized_gain_pct_norm | float64 | 0.7903270114834069 | 0.8025272676737958 | 0.7919912098405154 |

## Allowed state columns (v1)

- `sma_5`
- `ema_5`
- `sma_10`
- `ema_10`
- `sma_20`
- `ema_20`
- `sma_50`
- `ema_50`
- `rsi_14`
- `macd`
- `macd_signal`
- `macd_hist`
- `bb_lower`
- `bb_middle`
- `bb_upper`
- `Gold_Close`
- `WTI_Close`
- `Semiconductor_Close`
- `EUR_USD_Close`
- `USD_CNY_Close`
- `USD_JPN_Close`
- `Wheat_Close`
- `SPY_Close`
- `VIX_Close`
- `PC1`
- `PC2`
- `PC3`
- `PC4`
- `PC5`
- `PC6`
- `PC7`
- `PC8`
- `PC9`
- `PC10`
- `days_until_tax_transition`
- `unrealized_gains_pct`
- `days_to_tax_transition_norm`
- `unrealized_gain_pct_norm`

## Excluded columns (v1)

- `date`: raw temporal metadata; excluded in v1 for conservative state freeze
- `ticker`: identifier excluded in v1 to avoid asset-specific memorization; may be encoded later if needed
- `open`: raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables
- `high`: raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables
- `low`: raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables
- `close`: raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables
- `adj_close`: raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables
- `volume`: raw OHLCV excluded in v1; policy uses richer engineered features and tax-aware state variables
- `source`: raw metadata (universe/source provenance), not direct state input in v1
- `ticker_row`: simulator/bookkeeping metadata
- `simulated_purchase_price`: tax/construction variable excluded in v1 to avoid redundant shortcut state
- `simulated_purchase_pos`: simulator/bookkeeping metadata
- `simulated_purchase_date`: raw temporal metadata; excluded in v1 for conservative state freeze
- `trigger_candidate`: leakage risk: episode-construction trigger flag
- `valid_trigger`: leakage risk: episode-construction trigger flag
- `episode_id`: identifier
- `trigger_date`: raw temporal metadata; excluded in v1 for conservative state freeze
- `tax_transition_date`: raw temporal metadata; excluded in v1 for conservative state freeze
- `holding_period_days`: tax/construction variable excluded in v1 to avoid redundant shortcut state

## Rationale (short)

- Included: technical indicators, macro close series, PCA factors, and core tax-aware state (raw + normalized).
- Tax-aware columns in scope: `unrealized_gains_pct`, `days_until_tax_transition`, `unrealized_gain_pct_norm`, `days_to_tax_transition_norm`.
- Normalized tax mappings used upstream: `days_to_tax_transition_norm = min(days_until_tax_transition, 365) / 365`, `unrealized_gain_pct_norm = tanh(unrealized_gains_pct / 0.25)`.
- Excluded: OHLCV, identifiers, dates, trigger flags, simulator bookkeeping, source metadata, and `holding_period_days`.
- This freeze is intentionally conservative and only defines **v1** state visibility.

## Note

- This is a state freeze snapshot (v1), not a final modeling decision.