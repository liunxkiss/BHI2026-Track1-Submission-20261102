from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import numpy as np
RUNS = Path('/workspace/pulse-training/runs')
DATA_ROOT = RUNS / 'BHI_full900_train_balanced_audit_20261030_v2'
CAL_ROOT = RUNS / 'BHI_full900_six_experts_calibrated_20261031'
FINAL_ROOT = RUNS / 'BHI_full900_six_experts_calibrated_final_20261031'
OUTPUT = RUNS / 'BHI_full900_six_experts_router_ready_20261031'
TARGETS = ('SBP', 'DBP')
BANDS = ('low', 'normal', 'high')
SELECTED_Q = {'SBP_low': (0.55, 0.45, 0.0), 'SBP_normal': (0.025, 0.95, 0.025), 'SBP_high': (0.0, 0.18, 0.82), 'DBP_low': (0.984, 0.016, 0.0), 'DBP_normal': (0.1, 0.8, 0.1), 'DBP_high': (0.0, 0.3, 0.7)}

def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}

def write_json(path: Path, value: object) -> None:

    def default(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item))
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=default) + '\n', encoding='utf-8')
    os.replace(temp, path)

def band_mae(truth: np.ndarray, prediction: np.ndarray, bands: np.ndarray) -> dict:
    return {name: float(np.mean(np.abs(prediction[bands == index] - truth[bands == index]))) for index, name in enumerate(BANDS)}

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    train = load_archive(DATA_ROOT / 'TRAIN_FULL899.npz')
    labels = np.asarray(train['y'], dtype=np.float64)
    bands = np.asarray(train['bands'], dtype=np.int64)
    groups = np.asarray(train['groups'], dtype=str)
    indices = np.asarray(train['indices'], dtype=np.int64)
    final_oof = np.asarray(load_archive(FINAL_ROOT / 'EXPERT_OOF.npz')['oof'], dtype=np.float64)
    coarse_oof = np.asarray(load_archive(CAL_ROOT / 'EXPERT_OOF.npz')['oof'], dtype=np.float64)
    selected_oof = final_oof.copy()
    selected_oof[:, 0, 0] = coarse_oof[:, 0, 0]
    coarse_receipt = json.loads((CAL_ROOT / 'RESULTS.json').read_text(encoding='utf-8'))
    final_receipt = json.loads((FINAL_ROOT / 'RESULTS.json').read_text(encoding='utf-8'))
    model_sources = {'SBP_low': Path(coarse_receipt['selected_roles']['SBP_low']['full_fit']['model_path'])}
    for role_id in SELECTED_Q:
        if role_id != 'SBP_low':
            model_sources[role_id] = Path(final_receipt['selected'][role_id]['model_path'])
    model_receipts = {}
    for role_id, source in model_sources.items():
        bundle = joblib.load(source)
        expected_q = np.asarray(SELECTED_Q[role_id], dtype=np.float64)
        source_q = np.asarray(bundle['q'], dtype=np.float64)
        if not np.allclose(source_q, expected_q, atol=1e-12):
            raise AssertionError((role_id, source_q, expected_q))
        destination = OUTPUT / f'{role_id}_FULL.joblib'
        joblib.dump(bundle, destination, compress=3)
        model_receipts[role_id] = {'q': expected_q.tolist(), 'source_path': str(source), 'path': str(destination), 'sha256': sha256_file(destination), 'fit_rows': int(bundle.get('fit_rows', -1)), 'fit_groups': int(bundle.get('fit_groups', -1))}
    np.savez_compressed(OUTPUT / 'EXPERT_OOF.npz', oof=selected_oof, labels=labels, bands=bands, groups=groups, indices=indices, targets=np.asarray(TARGETS), experts=np.asarray(BANDS))
    audit_results = {}
    for target_index, target in enumerate(TARGETS):
        audit = load_archive(DATA_ROOT / f'VALIDATION_{target}_OVERLAP.npz')
        audit_indices = np.asarray(audit['indices'], dtype=np.int64)
        audit_truth = np.asarray(audit['y'][:, target_index], dtype=np.float64)
        audit_bands = np.asarray(audit['bands'][:, target_index], dtype=np.int64)
        target_results = {}
        for expert_index, expert in enumerate(BANDS):
            role_id = f'{target}_{expert}'
            prediction = selected_oof[audit_indices, expert_index, target_index]
            target_results[role_id] = band_mae(audit_truth, prediction, audit_bands)
        audit_results[target] = target_results
    receipt = {'schema': 'BHI_ROUTER_READY_SIX_EXPERTS_V1', 'status': 'PASS', 'records': int(len(labels)), 'groups': int(len(np.unique(groups))), 'weights': SELECTED_Q, 'weight_sums': {role_id: float(sum(q)) for role_id, q in SELECTED_Q.items()}, 'models': model_receipts, 'audit_oof_band_mae': audit_results, 'source_priority': {'SBP_low': 'low > normal; high excluded', 'SBP_normal': 'normal > low and high', 'SBP_high': 'high > normal; low excluded', 'DBP_low': 'low > normal; high excluded', 'DBP_normal': 'normal > low and high', 'DBP_high': 'high > normal; low excluded'}}
    receipt['files'] = {path.name: sha256_file(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    write_json(OUTPUT / 'RESULTS.json', receipt)
    print(json.dumps({'status': 'PASS', 'output': str(OUTPUT), 'weights': SELECTED_Q, 'audit_oof_band_mae': audit_results}, ensure_ascii=False))
if __name__ == '__main__':
    main()
