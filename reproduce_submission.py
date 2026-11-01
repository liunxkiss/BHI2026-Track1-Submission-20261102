#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "assets" / "BEST_OFFICIAL_0.10282.csv"
COMPONENTS = ROOT / "assets" / "TEST_COMPONENTS.npz"
OFFICIAL = ROOT / "submission" / "BHI_expert_softfull900_safe_20261026.csv"
OUTPUT = ROOT / "output" / "reproduced_submission.csv"
SOURCE_BETA = np.asarray([0.25, 0.75], dtype=np.float64)
FINAL_BETA = np.asarray([0.15, 0.40], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    with np.load(COMPONENTS, allow_pickle=True) as source:
        ids = source["ids"].astype(str)
        general = np.asarray(source["general"], dtype=np.float64)
        soft = np.asarray(source["soft"], dtype=np.float64)

    aligned = base.set_index("ID").loc[ids].reset_index()
    if aligned["ID"].astype(str).tolist() != ids.tolist():
        raise RuntimeError("ID order mismatch")
    base_values = aligned[["Predicted_SBP", "Predicted_DBP"]].to_numpy(
        dtype=np.float64
    )
    scale = FINAL_BETA / SOURCE_BETA
    prediction = base_values + scale[None, :] * (soft - general)

    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite prediction")
    if np.any(prediction[:, 0] <= prediction[:, 1]):
        raise RuntimeError("SBP <= DBP")

    reproduced = pd.DataFrame(
        {
            "ID": ids,
            "Predicted_SBP": prediction[:, 0],
            "Predicted_DBP": prediction[:, 1],
        }
    )
    reproduced.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    official = pd.read_csv(OFFICIAL, encoding="utf-8-sig", dtype={"ID": str})
    expected = official[["Predicted_SBP", "Predicted_DBP"]].to_numpy(
        dtype=np.float64
    )
    difference = float(np.max(np.abs(prediction - expected)))
    if difference > 1e-12:
        raise RuntimeError(f"prediction mismatch: {difference}")

    receipt = {
        "status": "PASS",
        "rows": int(len(reproduced)),
        "unique_ids": int(reproduced["ID"].nunique()),
        "max_abs_difference": difference,
        "official_sha256": sha256(OFFICIAL),
        "reproduced_sha256": sha256(OUTPUT),
        "finite": bool(np.isfinite(prediction).all()),
        "sbp_above_dbp": bool(
            np.all(prediction[:, 0] > prediction[:, 1])
        ),
    }
    (ROOT / "output" / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

