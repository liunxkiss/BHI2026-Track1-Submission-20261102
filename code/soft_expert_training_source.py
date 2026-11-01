from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
RUNS = Path('/workspace/pulse-training/runs')
DATA = RUNS / 'BHI_unified_bias_20261013_v3/data'
EXPERT_RUN = RUNS / 'BHI纯分段三专家_20261024'
PIPELINE = EXPERT_RUN / 'deployment/pipeline.joblib'
VALIDATION = EXPERT_RUN / '03_逐记录证据/evaluation/validation_predictions.csv'
PREDICT_SOURCE = EXPERT_RUN / '02_代码配置/predict_standalone.py'
TOP6 = RUNS / 'BHI_DEEP_RESEARCH_ROUTES_20261026' / 'BHI_public_top6_mean_20261026.csv'
OUT = RUNS / 'BHI_full900_soft_endpoint_20261026'
TRAIN_ROWS = 3620
TOTAL_ROWS = 4021
THREADS = 4
BETAS = {'SBP': 0.25, 'DBP': 0.75}
CUTS = {'SBP': (100.0, 140.0), 'DBP': (60.0, 90.0)}
TARGETS = ('SBP', 'DBP')
EXPERTS = ('low', 'normal', 'high')
CLASSIFIER_PARAMS = {'loss': 'log_loss', 'max_iter': 100, 'learning_rate': 0.05, 'max_leaf_nodes': 7, 'min_samples_leaf': 30, 'l2_regularization': 1.0, 'early_stopping': False, 'random_state': 42}

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def band_code(values: np.ndarray, target: str) -> np.ndarray:
    return np.digitize(values, CUTS[target])

def target_metrics(prediction: np.ndarray, labels: np.ndarray, target: str) -> dict:
    error = np.abs(prediction - labels)
    bias = prediction - labels
    codes = band_code(labels, target)
    return {'mae': float(error.mean()), 'bias': float(bias.mean()), 'segments': {expert: {'records': int((codes == index).sum()), 'mae': float(error[codes == index].mean()), 'bias': float(bias[codes == index].mean())} for index, expert in enumerate(EXPERTS)}}

def target_weights(duplicate: np.ndarray, groups: np.ndarray, indices: np.ndarray, mode: str) -> np.ndarray:
    selected_groups = groups[indices]
    output = duplicate[indices].astype(np.float64).copy()
    if mode == 'group':
        for group in np.unique(selected_groups):
            local = np.flatnonzero(selected_groups == group)
            mass = float(output[local].sum())
            if mass <= 0:
                raise RuntimeError(f'nonpositive group mass: {group}')
            output[local] /= mass
    output /= output.mean()
    return output

def load_oof_components(predict_module, features: np.ndarray, records: pd.DataFrame) -> dict:
    detail = pd.read_csv(VALIDATION, encoding='utf-8-sig')
    if len(detail) != TRAIN_ROWS:
        raise RuntimeError('OOF detail row count mismatch')
    by_id = detail.set_index('record_id', drop=False)
    ids = records.iloc[:TRAIN_ROWS]['segment_id'].astype(str).tolist()
    if set(ids) != set(by_id.index.astype(str)):
        raise RuntimeError('OOF record IDs mismatch')
    ordered = by_id.loc[ids]
    general = ordered[['general_SBP', 'general_DBP']].to_numpy(dtype=np.float64)
    experts = np.empty((TRAIN_ROWS, 3, 2), dtype=np.float64)
    for expert_index, expert in enumerate(EXPERTS):
        for target_index, target in enumerate(TARGETS):
            experts[:, expert_index, target_index] = ordered[f'new_candidate_{expert}_{target}'].to_numpy(dtype=np.float64)
    labels = ordered[['true_SBP', 'true_DBP']].to_numpy(dtype=np.float64)
    coverage = ordered['outer'].to_numpy(dtype=np.int64)
    return {'ids': np.asarray(ids, dtype=object), 'general': general, 'experts': experts, 'labels': labels, 'outer': coverage, 'matrix': predict_module.route_matrix(np.asarray(features[:TRAIN_ROWS], dtype=np.float64), general, experts)}

def fit_endpoint_gates(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray, duplicate: np.ndarray, mode: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, HistGradientBoostingClassifier]]]:
    oof = {target: {endpoint: np.full(TRAIN_ROWS, np.nan, dtype=np.float64) for endpoint in ('low', 'high')} for target in TARGETS}
    splitter = GroupKFold(n_splits=3)
    indices = np.arange(TRAIN_ROWS, dtype=np.int64)
    for fold, (fit, query) in enumerate(splitter.split(indices, groups=groups)):
        weights = target_weights(duplicate, groups, fit, mode)
        for target_index, target in enumerate(TARGETS):
            codes = band_code(labels[fit, target_index], target)
            for endpoint, destination in (('low', 0), ('high', 2)):
                model = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
                model.fit(matrix[fit], (codes == destination).astype(np.int8), sample_weight=weights)
                oof[target][endpoint][query] = model.predict_proba(matrix[query])[:, 1]
        print(json.dumps({'event': 'gate_fold_complete', 'mode': mode, 'fold': fold}), flush=True)
    if not all((np.isfinite(value).all() for target in oof.values() for value in target.values())):
        raise RuntimeError('incomplete endpoint gate OOF')
    final_models = {}
    all_weights = target_weights(duplicate, groups, indices, mode)
    for target_index, target in enumerate(TARGETS):
        final_models[target] = {}
        codes = band_code(labels[:, target_index], target)
        for endpoint, destination in (('low', 0), ('high', 2)):
            model = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
            model.fit(matrix, (codes == destination).astype(np.int8), sample_weight=all_weights)
            final_models[target][endpoint] = model
    return (oof, final_models)

def soft_prediction(general: np.ndarray, experts: np.ndarray, low_probability: np.ndarray, high_probability: np.ndarray, target_index: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = 1.0 + low_probability + high_probability
    low_weight = beta * low_probability / denominator
    high_weight = beta * high_probability / denominator
    prediction = general[:, target_index] + low_weight * (experts[:, 0, target_index] - general[:, target_index]) + high_weight * (experts[:, 2, target_index] - general[:, target_index])
    return (prediction, low_weight, high_weight)

def distribution(prediction: np.ndarray) -> dict:
    output = {}
    for target_index, target in enumerate(TARGETS):
        values = prediction[:, target_index]
        cutoff = CUTS[target]
        output[target] = {'minimum': float(values.min()), 'mean': float(values.mean()), 'std': float(values.std(ddof=0)), 'maximum': float(values.max()), 'bands': {name: int((np.digitize(values, cutoff) == index).sum()) for index, name in enumerate(EXPERTS)}}
    return output

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--threads', type=int, default=THREADS)
    args = parser.parse_args()
    if os.getuid() != 1003:
        raise RuntimeError(f'expected uid 1003, got {os.getuid()}')
    resource.setrlimit(resource.RLIMIT_AS, (48 * 1024 ** 3, 48 * 1024 ** 3))
    args.out.mkdir(parents=True, exist_ok=True)
    predict_module = load_module('bhi_pure_predict', PREDICT_SOURCE)
    features = np.load(DATA / 'new_features.npy', mmap_mode='r', allow_pickle=False)
    labels_file = np.load(DATA / 'labels.npy', allow_pickle=False).astype(np.float64)
    duplicate = np.load(DATA / 'duplicate_weights.npy', allow_pickle=False).astype(np.float64)
    records = pd.read_csv(DATA / 'records.csv', dtype=str)
    groups = records.iloc[:TRAIN_ROWS]['canonical_group'].to_numpy(dtype=str)
    oof = load_oof_components(predict_module, features, records)
    if not np.allclose(oof['labels'], labels_file[:TRAIN_ROWS], rtol=0, atol=0):
        raise RuntimeError('OOF labels do not match canonical label file')
    started = time.monotonic()
    all_oof = {}
    all_models = {}
    with threadpool_limits(args.threads):
        for mode in ('duplicate', 'group'):
            probabilities, models = fit_endpoint_gates(oof['matrix'], oof['labels'], groups, duplicate, mode)
            all_oof[mode] = probabilities
            all_models[mode] = models
    oof_low = {target: (all_oof['duplicate'][target]['low'] + all_oof['group'][target]['low']) / 2.0 for target in TARGETS}
    oof_high = {target: (all_oof['duplicate'][target]['high'] + all_oof['group'][target]['high']) / 2.0 for target in TARGETS}
    oof_soft = np.empty((TRAIN_ROWS, 2), dtype=np.float64)
    oof_low_weight = np.empty_like(oof_soft)
    oof_high_weight = np.empty_like(oof_soft)
    for target_index, target in enumerate(TARGETS):
        prediction, low_weight, high_weight = soft_prediction(oof['general'], oof['experts'], oof_low[target], oof_high[target], target_index, BETAS[target])
        oof_soft[:, target_index] = prediction
        oof_low_weight[:, target_index] = low_weight
        oof_high_weight[:, target_index] = high_weight
    oof_general_metrics = {target: target_metrics(oof['general'][:, index], oof['labels'][:, index], target) for index, target in enumerate(TARGETS)}
    oof_soft_metrics = {target: target_metrics(oof_soft[:, index], oof['labels'][:, index], target) for index, target in enumerate(TARGETS)}
    oof_general_weighted = float(0.6 * oof_general_metrics['SBP']['mae'] + 0.4 * oof_general_metrics['DBP']['mae'])
    oof_soft_weighted = float(0.6 * oof_soft_metrics['SBP']['mae'] + 0.4 * oof_soft_metrics['DBP']['mae'])
    test = predict_module.predict(PIPELINE, features[TRAIN_ROWS:TOTAL_ROWS], workspace_root=EXPERT_RUN)
    test_matrix = predict_module.route_matrix(np.asarray(features[TRAIN_ROWS:TOTAL_ROWS], dtype=np.float64), test['general'], test['experts'])
    test_low = {}
    test_high = {}
    for target in TARGETS:
        test_low[target] = (all_models['duplicate'][target]['low'].predict_proba(test_matrix)[:, 1] + all_models['group'][target]['low'].predict_proba(test_matrix)[:, 1]) / 2.0
        test_high[target] = (all_models['duplicate'][target]['high'].predict_proba(test_matrix)[:, 1] + all_models['group'][target]['high'].predict_proba(test_matrix)[:, 1]) / 2.0
    test_soft = np.empty((TOTAL_ROWS - TRAIN_ROWS, 2), dtype=np.float64)
    test_low_weight = np.empty_like(test_soft)
    test_high_weight = np.empty_like(test_soft)
    for target_index, target in enumerate(TARGETS):
        prediction, low_weight, high_weight = soft_prediction(test['general'], test['experts'], test_low[target], test_high[target], target_index, BETAS[target])
        test_soft[:, target_index] = prediction
        test_low_weight[:, target_index] = low_weight
        test_high_weight[:, target_index] = high_weight
    invalid = test_soft[:, 0] <= test_soft[:, 1]
    fallback_count = int(invalid.sum())
    test_soft[invalid] = test['general'][invalid]
    ids = records.iloc[TRAIN_ROWS:TOTAL_ROWS]['segment_id'].astype(str).to_numpy()
    top6 = pd.read_csv(TOP6, encoding='utf-8-sig')
    if ids.tolist() != top6['ID'].astype(str).tolist():
        raise RuntimeError('test ID order mismatch')
    candidate = pd.DataFrame({'ID': ids, 'Predicted_SBP': test_soft[:, 0], 'Predicted_DBP': test_soft[:, 1]})
    csv_path = args.out / 'BHI_full900_soft_endpoint_20261026.csv'
    candidate.to_csv(csv_path, index=False, encoding='utf-8-sig')
    models_path = args.out / 'endpoint_gates.joblib'
    joblib.dump(all_models, models_path)
    np.savez_compressed(args.out / 'OOF_COMPONENTS.npz', train_indices=np.arange(TRAIN_ROWS, dtype=np.int64), labels=oof['labels'], general=oof['general'], experts=oof['experts'], low_probability_sbp=oof_low['SBP'], high_probability_sbp=oof_high['SBP'], low_probability_dbp=oof_low['DBP'], high_probability_dbp=oof_high['DBP'], soft=oof_soft)
    np.savez_compressed(args.out / 'TEST_COMPONENTS.npz', ids=ids, general=test['general'], experts=test['experts'], low_probability_sbp=test_low['SBP'], high_probability_sbp=test_high['SBP'], low_probability_dbp=test_low['DBP'], high_probability_dbp=test_high['DBP'], low_weight=test_low_weight, high_weight=test_high_weight, soft=test_soft, fallback_mask=invalid)
    manifest = {'schema': 'BHI_FULL900_SOFT_ENDPOINT_V1', 'training_records': TRAIN_ROWS, 'training_groups': int(len(set(groups))), 'test_records': int(len(ids)), 'betas': BETAS, 'gate_ensemble': 'average(duplicate-weight HGB, group-equal-weight HGB)', 'oof_general_weighted_mae': oof_general_weighted, 'oof_soft_weighted_mae': oof_soft_weighted, 'oof_improvement': float(oof_general_weighted - oof_soft_weighted), 'oof_general': oof_general_metrics, 'oof_soft': oof_soft_metrics, 'test_distribution': distribution(test_soft), 'top6_distribution': distribution(top6[['Predicted_SBP', 'Predicted_DBP']].to_numpy(dtype=np.float64)), 'physiology_fallback_records': fallback_count, 'csv_sha256': sha256(csv_path), 'models_sha256': sha256(models_path), 'threads': args.threads, 'seconds': float(time.monotonic() - started), 'test_labels_read': False, 'singleton_inference': True}
    (args.out / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
