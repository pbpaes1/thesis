# Brazil sensitivity state freeze v2

## Inventory audit and freeze decision

The v2 state contains 36 inputs: 35 unchanged v1 inputs and the new terminal-horizon input in the old countdown's position.
Remaining inventory and cash-lot history are known to the simulator but deliberately omitted from observations to preserve the 36-input comparison with U.S. C-lite v5, as chosen for this freeze.
The feedforward policy is therefore partially observable: identical market/time inputs can imply different remaining sale capacity and interest-tax exposure. No static inventory column is created.

## Data and validation

- Source: `data/episodes/drl_episodes.parquet`
- Scenario: `data/episodes/drl_episodes_brazil_v1.parquet`
- Rows: 973,658; episodes: 10,119; columns: 58; allowed: 36; excluded: 22.
- Source rows and columns retained in order; original parquet hash unchanged after generation.
- Episode dates strictly increase; episode/date pairs are unique; required prices, basis, and state inputs are finite and nonmissing.
- Terminal horizon uses each episode's actual last observation date, is in [0, 1], non-increasing, and zero on every terminal row.
- Old tax countdown remains in the scenario parquet for provenance but is excluded from observations; future prices and simulator accounting are not state inputs.
- Scenario IDs and chronological train/validation/test membership and ordering match the C-lite v5 control exactly.
- All six Brazil configurations reference this parquet and v2 allowed-state JSON.

## SHA-256

ID-list hashes use UTF-8 IDs joined by LF without a trailing newline. The universe list is sorted lexicographically; split lists retain control CSV order.
- Original parquet: `0adf5bcb50da30314b19c04db1ca1e122ca6f823087d4a8025a5ecbe6537570a`
- Scenario parquet: `28ae8ff389d0be2b493b426bd32822b2b36f642f172f2f7b418ce5187ce7e2ff`
- Sorted episode IDs: `5b51fb2fa129192a242ef1fc037dfed1d0ea7ef14d84e8951470808c6be57f4e`
- train: 7,083 IDs; ordered-list hash `e83c5717229acfa8c901fd22c7a192ef783ff4a8cc926396bedde7b49e5df7d4`
- validation: 1,517 IDs; ordered-list hash `af345e418b3668dde467c6f4287ceb940ae65fdcc8cc03cce61ee73d18120c6c`
- test: 1,519 IDs; ordered-list hash `273d2547e2cba607dbbb87a2f5125a304516caf70fd7269859968b30c2b6f98c`
