from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from analyze_old_new_fusion import TARGETS, confusion, endpoint_rescue, load_historical, load_new
from train_action_selector import compact_context, probability_features, sample_weights
OUT = Path('/workspace/pulse-training/runs/BHI_normal_rescue_20261101')

def model_for(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.05, max_iter=100, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1.0, early_stopping=False, random_state=seed)

def run_target(target: str) -> None:
    truth, groups, old_probability, old_choice = load_historical(target)
    new_choice, new_probability = load_new(target)
    base_choice = endpoint_rescue(old_probability, old_choice, new_probability, new_choice, 0.6, 0.2)
    context = compact_context(target)
    features = np.column_stack((probability_features(old_probability, old_choice), probability_features(new_probability, new_choice), old_probability - new_probability, context)).astype(np.float32)
    endpoint = base_choice != 1
    labels = ((truth == 1) & endpoint).astype(np.int64)
    base_weights = sample_weights(groups, truth)
    reports = {}
    for mode in ('natural', 'balanced'):
        weights = base_weights.copy()
        if mode == 'balanced':
            class_weights = {value: len(labels[endpoint]) / (2.0 * max(int((labels[endpoint] == value).sum()), 1)) for value in (0, 1)}
            local = np.asarray([class_weights[int(label)] for label in labels[endpoint]])
            weights[endpoint] *= local
            weights /= weights.mean()
        oof_probability = np.zeros(len(truth), dtype=np.float64)
        splitter = GroupKFold(n_splits=5)
        for fold, (fit, held) in enumerate(splitter.split(features, labels, groups)):
            local_fit = fit[endpoint[fit]]
            local_held = held[endpoint[held]]
            model = model_for(20261201 + TARGETS.index(target) * 1000 + fold)
            model.fit(features[local_fit], labels[local_fit], sample_weight=weights[local_fit])
            oof_probability[local_held] = model.predict_proba(features[local_held])[:, 1]
        policies = {}
        for threshold in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.9):
            choice = base_choice.copy()
            rescue = endpoint & (oof_probability >= threshold)
            choice[rescue] = 1
            policies[f'rescue_{threshold:.2f}'] = {**confusion(truth, choice), 'rescued_records': int(rescue.sum())}
        full_model = model_for(20261201 + TARGETS.index(target) * 1000 + 50)
        full_fit = np.flatnonzero(endpoint)
        full_model.fit(features[full_fit], labels[full_fit], sample_weight=weights[full_fit])
        output = OUT / target
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': full_model, 'target': target, 'feature_count': features.shape[1], 'mode': mode}, output / f'NORMAL_RESCUE_{mode}.joblib', compress=3)
        np.savez_compressed(output / f'NORMAL_RESCUE_OOF_{mode}.npz', groups=groups, truth=truth, base_choice=base_choice, endpoint=endpoint, probability=oof_probability)
        reports[mode] = policies
    report = {'schema': 'BHI_NORMAL_RESCUE_V1', 'target': target, 'base': confusion(truth, base_choice), 'endpoint_records': int(endpoint.sum()), 'normal_positive_records': int(labels.sum()), 'modes': reports}
    output = OUT / target
    output.mkdir(parents=True, exist_ok=True)
    (output / 'RESULTS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    target = sys.argv[1]
    if target not in TARGETS:
        raise ValueError(target)
    run_target(target)
