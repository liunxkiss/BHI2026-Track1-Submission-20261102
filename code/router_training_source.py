from __future__ import annotations
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
RUNS = Path('/workspace/pulse-training/runs')
MATERIAL_ROOT = RUNS / 'BHI_router_ready_material_20261031_v2'
TRAIN_MATERIAL = MATERIAL_ROOT / 'ROUTER_TRAIN_MATERIAL.npz'
OUTPUT = RUNS / 'BHI_router_expected_error_full900_20261031'
TARGETS = ('SBP', 'DBP')
EXPERTS = ('low', 'normal', 'high')
N_FOLDS = 5
ROUTER_PARAMS = {'loss': 'squared_error', 'learning_rate': 0.05, 'max_iter': 100, 'max_leaf_nodes': 7, 'min_samples_leaf': 30, 'l2_regularization': 1.0, 'early_stopping': False, 'max_bins': 127, 'random_state': 42}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()

def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))

def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + '\n', encoding='utf-8')
    os.replace(temp, path)

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)

def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}

def group_class_weights(bands: np.ndarray, groups: np.ndarray, duplicate: np.ndarray) -> tuple[np.ndarray, dict]:
    unique_bands = np.unique(bands)
    if not np.array_equal(unique_bands, np.arange(3)):
        raise AssertionError(('missing router train class', unique_bands))
    weights = np.zeros(len(bands), dtype=np.float64)
    group_counts = []
    class_totals = []
    for band_index in range(3):
        local = np.flatnonzero(bands == band_index)
        local_groups = np.unique(groups[local])
        group_counts.append(int(len(local_groups)))
        class_mass = 0.0
        for group in local_groups:
            mask = (bands == band_index) & (groups == group)
            group_duplicate = duplicate[mask]
            mass = float(group_duplicate.sum())
            if mass <= 0:
                raise AssertionError(('bad duplicate mass', band_index, group))
            weights[mask] = group_duplicate / (len(local_groups) * mass) / 3
            class_mass += float(weights[mask].sum())
        class_totals.append(class_mass)
    if not np.allclose(class_totals, [1 / 3] * 3, atol=1e-12):
        raise AssertionError(('class weight totals', class_totals))
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError(('weight sum', weights.sum()))
    return (weights, {'class_totals': class_totals, 'class_groups': group_counts, 'weight_sum': float(weights.sum()), 'weight_min': float(weights.min()), 'weight_median': float(np.median(weights)), 'weight_max': float(weights.max()), 'weight_sha256': array_hash(weights)})

def confusion(rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (rows, columns), 1)
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    return (matrix, row_totals, column_totals)

def classification_metrics(rows: np.ndarray, columns: np.ndarray) -> dict:
    matrix, row_totals, column_totals = confusion(rows, columns)
    recall = np.divide(np.diag(matrix), row_totals, out=np.zeros(3, dtype=np.float64), where=row_totals > 0)
    precision = np.divide(np.diag(matrix), column_totals, out=np.zeros(3, dtype=np.float64), where=column_totals > 0)
    return {'matrix': matrix, 'row_totals': row_totals, 'column_totals': column_totals, 'recall': recall, 'precision': precision, 'accuracy': float(np.trace(matrix) / matrix.sum()), 'macro_recall': float(recall.mean()), 'minimum_recall': float(recall.min())}

def selected_metrics(candidate_error: np.ndarray, choice: np.ndarray) -> dict:
    selected_error = candidate_error[np.arange(len(choice)), choice]
    oracle_error = candidate_error.min(axis=1)
    regret = selected_error - oracle_error
    return {'selected_mae': float(selected_error.mean()), 'oracle_mae': float(oracle_error.mean()), 'mean_regret': float(regret.mean()), 'selected_error_p90': float(np.quantile(selected_error, 0.9)), 'regret_p90': float(np.quantile(regret, 0.9))}

def cell_metrics(row: int, column: int, bands: np.ndarray, choice: np.ndarray, candidate_error: np.ndarray) -> dict:
    mask = (bands == row) & (choice == column)
    if not np.any(mask):
        return {'records': 0, 'row_fraction': None, 'selected_mae': None, 'regret_mean': None, 'regret_sum': None}
    row_count = int(np.sum(bands == row))
    selected = np.take_along_axis(candidate_error, choice[:, None], axis=1)[:, 0]
    oracle = candidate_error.min(axis=1)
    regret = selected - oracle
    return {'records': int(mask.sum()), 'row_fraction': float(mask.sum() / row_count), 'selected_mae': float(selected[mask].mean()), 'regret_mean': float(regret[mask].mean()), 'regret_sum': float(regret[mask].sum())}

def build_matrix_rows(row_labels: np.ndarray, choice: np.ndarray, groups: np.ndarray, candidate_error: np.ndarray, row_basis: str) -> tuple[list[dict], dict]:
    metrics = classification_metrics(row_labels, choice)
    rows = []
    for row in range(3):
        for column in range(3):
            cell = cell_metrics(row, column, row_labels, choice, candidate_error)
            rows.append({'row_basis': row_basis, 'true_or_actual_best': EXPERTS[row], 'selected_expert': EXPERTS[column], **cell})
    metrics['row_groups'] = [int(len(np.unique(groups[row_labels == row]))) for row in range(3)]
    return (rows, metrics)

def detailed_confusion(true_band: np.ndarray, choice: np.ndarray, groups: np.ndarray, candidate_error: np.ndarray) -> tuple[list[dict], dict]:
    base = classification_metrics(true_band, choice)
    outcomes = []
    selected = np.take_along_axis(candidate_error, choice[:, None], axis=1)[:, 0]
    oracle = candidate_error.min(axis=1)
    regret = selected - oracle
    for row in range(3):
        for column in range(3):
            mask = (true_band == row) & (choice == column)
            outcomes.append({'true_band': EXPERTS[row], 'selected_expert': EXPERTS[column], 'records': int(mask.sum()), 'row_fraction': float(mask.sum() / np.sum(true_band == row)) if np.any(true_band == row) else None, 'groups': int(len(np.unique(groups[mask])) if np.any(mask) else np.zeros(0)), 'selected_mae': float(selected[mask].mean()) if np.any(mask) else None, 'regret_mean': float(regret[mask].mean()) if np.any(mask) else None, 'regret_sum': float(regret[mask].sum()) if np.any(mask) else None})
    base['row_groups'] = [int(len(np.unique(groups[true_band == row]))) for row in range(3)]
    return (outcomes, base)

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    material = load_archive(TRAIN_MATERIAL)
    x = np.asarray(material['X'], dtype=np.float32)
    y = np.asarray(material['y'], dtype=np.float64)
    bands = np.asarray(material['bands'], dtype=np.int64)
    groups = np.asarray(material['groups'], dtype=str)
    subjects = np.asarray(material['subjects'], dtype=str)
    indices = np.asarray(material['indices'], dtype=np.int64)
    record_ids = np.asarray(material['record_ids'], dtype=str)
    duplicate = np.asarray(material['duplicate_weight'], dtype=np.float64)
    fold_id = np.asarray(material['fold_id'], dtype=np.int64)
    candidate_error = np.asarray(material['candidate_error'], dtype=np.float64)
    candidate_values = np.asarray(material['expert_oof'], dtype=np.float64)
    general_values = np.asarray(material['general_oof'], dtype=np.float64)
    precondition = {'x_shape': x.shape == (3620, 1563), 'relation_features_finite': bool(np.isfinite(x[:, 1543:]).all()), 'raw_features_use_hgb_native_nan': True, 'y_shape': y.shape == (3620, 2), 'labels_finite': bool(np.isfinite(y).all()), 'groups_899': len(np.unique(groups)) == 899, 'subjects_900': len(np.unique(subjects)) == 900, 'indices_complete': np.array_equal(indices, np.arange(3620, dtype=np.int64)), 'fold_ids_complete': np.array_equal(np.unique(fold_id), np.arange(N_FOLDS)), 'candidate_values_shape': candidate_values.shape == (3620, 3, 2), 'candidate_values_finite': bool(np.isfinite(candidate_values).all()), 'general_values_shape': general_values.shape == (3620, 2), 'general_values_finite': bool(np.isfinite(general_values).all()), 'candidate_error_shape': candidate_error.shape == (3620, 3, 2), 'candidate_error_finite': bool(np.isfinite(candidate_error).all()), 'candidate_error_nonnegative': bool(np.all(candidate_error >= 0))}
    for target_index in range(2):
        expected_error = np.abs(candidate_values[:, :, target_index] - y[:, target_index, None])
        precondition[f'candidate_error_recomputed_{TARGETS[target_index]}'] = bool(np.array_equal(expected_error, candidate_error[:, :, target_index]))
    for fold in range(N_FOLDS):
        fold_groups = groups[fold_id == fold]
        precondition[f'fold_{fold}_group_unique'] = len(np.unique(fold_groups)) == len(fold_groups) // 4 or len(np.unique(fold_groups)) in (179, 180)
    if not all(precondition.values()):
        raise AssertionError(precondition)
    expected_error = np.full((len(x), 3, 2), np.nan, dtype=np.float64)
    successful_choice = np.full((len(x), 2), -1, dtype=np.int64)
    fold_receipts = []
    model_receipts = []
    total_started = time.monotonic()
    for fold in range(N_FOLDS):
        fit = np.flatnonzero(fold_id != fold)
        query = np.flatnonzero(fold_id == fold)
        if set(groups[fit]) & set(groups[query]):
            raise AssertionError(('group leakage', fold))
        fold_band_groups = {}
        for target_index, target in enumerate(TARGETS):
            weights, weight_receipt = group_class_weights(bands[fit, target_index], groups[fit], duplicate[fit])
            fold_band_groups[target] = weight_receipt
            for expert_index, expert in enumerate(EXPERTS):
                started = time.monotonic()
                model = HistGradientBoostingRegressor(**ROUTER_PARAMS)
                target_vector = candidate_error[fit, expert_index, target_index]
                model.fit(x[fit], target_vector, sample_weight=weights)
                prediction = np.asarray(model.predict(x[query]), dtype=np.float64)
                if prediction.shape != (len(query),):
                    raise AssertionError((target, expert, prediction.shape))
                if not np.isfinite(prediction).all():
                    raise AssertionError((target, expert, 'nonfinite'))
                expected_error[query, expert_index, target_index] = prediction
                model_path = OUTPUT / 'models' / f'fold_{fold}' / f'{target}_{expert}.joblib'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                bundle = {'schema': 'BHI_EXPECTED_ERROR_ROUTER_V2', 'target': target, 'target_index': target_index, 'expert': expert, 'expert_index': expert_index, 'fold': fold, 'model': model, 'parameters': ROUTER_PARAMS, 'input_dim': int(x.shape[1]), 'fit_records': int(len(fit)), 'fit_groups': int(len(np.unique(groups[fit]))), 'query_records': int(len(query)), 'query_groups': int(len(np.unique(groups[query]))), 'fit_input_sha256': array_hash(x[fit]), 'fit_target_sha256': array_hash(target_vector), 'fit_weight_sha256': weight_receipt['weight_sha256'], 'query_input_sha256': array_hash(x[query]), 'test_labels_read': False}
                joblib.dump(bundle, model_path, compress=3)
                model_receipts.append({'fold': fold, 'target': target, 'expert': expert, 'model_path': str(model_path), 'model_sha256': sha256_file(model_path), 'iterations': int(model.n_iter_), 'fit_records': int(len(fit)), 'fit_groups': int(len(np.unique(groups[fit]))), 'query_records': int(len(query)), 'query_groups': int(len(np.unique(groups[query]))), 'target_hash': array_hash(target_vector), 'weight_receipt': weight_receipt, 'fit_seconds': round(time.monotonic() - started, 3)})
                print(json.dumps({'event': 'fit_complete', 'fold': fold, 'target': target, 'expert': expert, 'seconds': round(time.monotonic() - started, 3)}), flush=True)
        for target_index, target in enumerate(TARGETS):
            clipped = np.maximum(expected_error[query, :, target_index], 0.0)
            choice = np.argmin(clipped, axis=1).astype(np.int64)
            successful_choice[query, target_index] = choice
            if np.any(successful_choice[query, target_index] < 0):
                raise AssertionError((target, 'missing choice'))
        fold_receipts.append({'fold': fold, 'fit_records': int(len(fit)), 'fit_groups': int(len(np.unique(groups[fit]))), 'query_records': int(len(query)), 'query_groups': int(len(np.unique(groups[query]))), 'fit_query_group_overlap': 0, 'class_weight_receipts': fold_band_groups})
    if not np.isfinite(expected_error).all():
        raise AssertionError('incomplete expected error OOF')
    if np.any(successful_choice < 0):
        raise AssertionError('incomplete OOF choices')
    replay_max_abs = 0.0
    for row in model_receipts:
        fold = int(row['fold'])
        target_index = TARGETS.index(row['target'])
        expert_index = EXPERTS.index(row['expert'])
        query = np.flatnonzero(fold_id == fold)
        bundle = joblib.load(row['model_path'])
        replay = np.asarray(bundle['model'].predict(x[query]), dtype=np.float64)
        replay_max_abs = max(replay_max_abs, float(np.max(np.abs(replay - expected_error[query, expert_index, target_index]))))
    if replay_max_abs != 0.0:
        raise AssertionError(replay_max_abs)
    np.savez_compressed(OUTPUT / 'ROUTER_OOF.npz', expected_error=expected_error, choice=successful_choice, candidate_values=candidate_values, candidate_error=candidate_error, general_values=general_values, y=y, bands=bands, groups=groups, subjects=subjects, indices=indices, record_ids=record_ids, fold_id=fold_id, targets=np.asarray(TARGETS), experts=np.asarray(EXPERTS))
    result = {'schema': 'BHI_EXPECTED_ERROR_ROUTER_FULL900_V1', 'status': 'PASS', 'scope': {'records': int(len(x)), 'groups': int(len(np.unique(groups))), 'subjects': int(len(np.unique(subjects))), 'features': int(x.shape[1]), 'folds': N_FOLDS, 'model_count': len(model_receipts), 'test_labels_read': False}, 'parameters': ROUTER_PARAMS, 'sample_weight': 'canonical_group_equal within true band; three true bands receive equal total weight', 'precondition_checks': precondition, 'fold_receipts': fold_receipts, 'model_receipts': model_receipts, 'replay_max_abs': replay_max_abs, 'all_oof': {}, 'audit_overlap': {}}
    record_rows = []
    for target_index, target in enumerate(TARGETS):
        actual_best = candidate_error[:, :, target_index].argmin(axis=1)
        selected_values = candidate_values[np.arange(len(x)), successful_choice[:, target_index], target_index]
        selected_error = candidate_error[np.arange(len(x)), successful_choice[:, target_index], target_index]
        oracle_error = candidate_error[:, :, target_index].min(axis=1)
        normal_error = candidate_error[:, 1, target_index]
        general_error = np.abs(general_values[:, target_index] - y[:, target_index])
        true_band_rows, true_band_metrics = detailed_confusion(bands[:, target_index], successful_choice[:, target_index], groups, candidate_error[:, :, target_index])
        actual_best_rows, actual_best_metrics = build_matrix_rows(actual_best, successful_choice[:, target_index], groups, candidate_error[:, :, target_index], 'actual_best')
        result['all_oof'][target] = {'true_band_confusion': true_band_metrics, 'true_band_cells': true_band_rows, 'actual_best_confusion': actual_best_metrics, 'actual_best_cells': actual_best_rows, 'selection_error': selected_metrics(candidate_error[:, :, target_index], successful_choice[:, target_index]), 'normal_candidate_mae': float(normal_error.mean()), 'general_model_mae': float(general_error.mean()), 'chosen_expert_counts': np.bincount(successful_choice[:, target_index], minlength=3), 'actual_best_counts': np.bincount(actual_best, minlength=3)}
        audit = load_archive(MATERIAL_ROOT / f'ROUTER_AUDIT_{target}_MATERIAL.npz')
        audit_indices = np.asarray(audit['indices'], dtype=np.int64)
        audit_bands = bands[audit_indices, target_index]
        audit_actual_best = candidate_error[audit_indices, :, target_index].argmin(axis=1)
        audit_choice = successful_choice[audit_indices, target_index]
        audit_groups = groups[audit_indices]
        audit_error_candidates = candidate_error[audit_indices, :, target_index]
        audit_true_rows, audit_true_metrics = detailed_confusion(audit_bands, audit_choice, audit_groups, audit_error_candidates)
        audit_best_rows, audit_best_metrics = build_matrix_rows(audit_actual_best, audit_choice, audit_groups, audit_error_candidates, 'actual_best')
        audit_selected_values = candidate_values[audit_indices, audit_choice, target_index]
        audit_selected_error = audit_error_candidates[np.arange(len(audit_indices)), audit_choice]
        audit_oracle_error = audit_error_candidates.min(axis=1)
        result['audit_overlap'][target] = {'records': int(len(audit_indices)), 'groups': int(len(np.unique(audit_groups))), 'true_band_confusion': audit_true_metrics, 'true_band_cells': audit_true_rows, 'actual_best_confusion': audit_best_metrics, 'actual_best_cells': audit_best_rows, 'selection_error': {'selected_mae': float(audit_selected_error.mean()), 'oracle_mae': float(audit_oracle_error.mean()), 'mean_regret': float((audit_selected_error - audit_oracle_error).mean())}, 'normal_candidate_mae': float(np.abs(candidate_values[audit_indices, 1, target_index] - y[audit_indices, target_index]).mean()), 'general_model_mae': float(np.abs(general_values[audit_indices, target_index] - y[audit_indices, target_index]).mean()), 'chosen_expert_counts': np.bincount(audit_choice, minlength=3), 'actual_best_counts': np.bincount(audit_actual_best, minlength=3)}
        for position, original_index in enumerate(audit_indices):
            record_rows.append({'target': target, 'record_id': str(record_ids[original_index]), 'index': int(original_index), 'group': str(groups[original_index]), 'subject_id': str(subjects[original_index]), 'fold': int(fold_id[original_index]), 'true_value': float(y[original_index, target_index]), 'true_band': EXPERTS[int(bands[original_index, target_index])], 'general_prediction': float(general_values[original_index, target_index]), 'low_prediction': float(candidate_values[original_index, 0, target_index]), 'normal_prediction': float(candidate_values[original_index, 1, target_index]), 'high_prediction': float(candidate_values[original_index, 2, target_index]), 'low_true_error': float(candidate_error[original_index, 0, target_index]), 'normal_true_error': float(candidate_error[original_index, 1, target_index]), 'high_true_error': float(candidate_error[original_index, 2, target_index]), 'expected_low_error': float(max(expected_error[original_index, 0, target_index], 0.0)), 'expected_normal_error': float(max(expected_error[original_index, 1, target_index], 0.0)), 'expected_high_error': float(max(expected_error[original_index, 2, target_index], 0.0)), 'selected_expert': EXPERTS[int(audit_choice[position])], 'actual_best_expert': EXPERTS[int(audit_actual_best[position])], 'selected_prediction': float(audit_selected_values[position]), 'selected_abs_error': float(audit_selected_error[position]), 'oracle_abs_error': float(audit_oracle_error[position])})
        write_csv(OUTPUT / f'{target}_TRUE_BAND_3X3.csv', audit_true_rows, ['true_band', 'selected_expert', 'records', 'row_fraction', 'groups', 'selected_mae', 'regret_mean', 'regret_sum'])
        write_csv(OUTPUT / f'{target}_ACTUAL_BEST_3X3.csv', audit_best_rows, ['row_basis', 'true_or_actual_best', 'selected_expert', 'records', 'row_fraction', 'selected_mae', 'regret_mean', 'regret_sum'])
    write_csv(OUTPUT / 'ROUTER_AUDIT_RECORD_PREDICTIONS.csv', record_rows, ['target', 'record_id', 'index', 'group', 'subject_id', 'fold', 'true_value', 'true_band', 'general_prediction', 'low_prediction', 'normal_prediction', 'high_prediction', 'low_true_error', 'normal_true_error', 'high_true_error', 'expected_low_error', 'expected_normal_error', 'expected_high_error', 'selected_expert', 'actual_best_expert', 'selected_prediction', 'selected_abs_error', 'oracle_abs_error'])
    result['elapsed_seconds'] = round(time.monotonic() - total_started, 3)
    result['input_files'] = {'material': str(TRAIN_MATERIAL), 'material_sha256': sha256_file(TRAIN_MATERIAL)}
    result['output_files'] = {path.name: sha256_file(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}
    write_json(OUTPUT / 'RESULTS.json', result)
    print(json.dumps({'status': 'PASS', 'output': str(OUTPUT), 'audit_true_band_recall': {target: result['audit_overlap'][target]['true_band_confusion']['recall'] for target in TARGETS}, 'audit_actual_best_recall': {target: result['audit_overlap'][target]['actual_best_confusion']['recall'] for target in TARGETS}, 'audit_selected_mae': {target: result['audit_overlap'][target]['selection_error']['selected_mae'] for target in TARGETS}}, ensure_ascii=False, default=json_default))
if __name__ == '__main__':
    main()
