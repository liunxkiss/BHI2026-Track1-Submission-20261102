# BHI 2026 Track 1 Submission Package

Team: `haha_kiss`

Contact: `lin5891396lin@gmail.com`

Track: Track 1

Method: endpoint router with low, normal, and high expert models

Public score: `0.10285`

Submission:

`submission/BHI_expert_softfull900_safe_20261026.csv`

SHA-256:

`dc053e527b53ce7a4ed158e4c4c1fb4340c8e41972a081524f8f58c07a9be492`

## Reproduce

```bash
python -m pip install -r requirements.txt
python reproduce_submission.py
```

The script rebuilds the 401-row CSV from the included frozen base and expert
components. It verifies all IDs, finite values, SBP > DBP, and the submission
checksum.

## Method

The submitted model uses endpoint gates to blend low and high expert residuals
into a protected full-range prediction. SBP and DBP use separate gate
probabilities and blend coefficients.

The package also contains the later structured router and canonical six-expert
models:

- SBP and DBP are routed independently.
- Group-isolated folds keep all records from one canonical subject together.
- Expert weights sum to one for each of the six roles.
- Both the 3,077-record balanced router and the 3,620-record expert artifacts
  are included.

## Files

- `submission/`: submitted CSV.
- `assets/`: frozen base and expert component arrays.
- `config/`: public score receipt, construction manifest, team information.
- `code/`: endpoint expert, structured router, normal rescue, and expert
  consolidation source.
- `models/`: router-ready expert weights and structured router checkpoints.
- `checksums.sha256`: file integrity checks.

## Compliance

No official test labels or external labels are read. Reproduction runs offline.
No external pretrained weights are required by the reproduction script. Training
source uses only the official challenge feature and label files.

