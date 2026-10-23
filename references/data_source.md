# Data source: IBM Telco Customer Churn snapshot

## Identity of the file we use

| Field | Value |
|---|---|
| Local path | `data/raw/telco.csv` |
| Size | 970 457 bytes |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

## How to obtain the same bytes

Download the commit-pinned CSV from the IBM source repository:

```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv
```

Then verify the hash before using it:

```sh
sha256sum data/raw/telco.csv
```

A different hash means the bytes differ: do not treat that file as the original
snapshot. This commit-pinned URL guarantees that every teammate downloads
exactly the same version of the file.

## Handling rules

- The file lives in `data/raw/` and is **immutable**: nothing in the project
  rewrites, reformats or renames it. Do not open and re-save it from a
  spreadsheet application (that would silently change its byte identity).
- The raw CSV is kept out of Git (see `.gitignore` and the README policy);
  the pinned URL + SHA-256 above are the reproduction contract.

## Field notes

- 7 043 customers (rows), 21 columns.
- `customerID` — unique customer identifier; must never be a model feature.
- `Churn` — target, labels `Yes` / `No`. Imbalanced: about 26% `Yes`.
- `tenure` (integer months), `MonthlyCharges` (float), `TotalCharges`
  (read as **text** because 11 rows contain a blank/whitespace value instead
  of a number — see `reports/data_overview.json`).
- The remaining 15 columns are categorical service/billing attributes
  (e.g. `Contract`, `InternetService`, `PaymentMethod`).
- This is a single **customer snapshot**, not an event history: all rows
  describe the same moment, so no time-based leakage control is possible.
