from __future__ import annotations
import argparse
import copy
import json
import os
import random
import re
import time
from pathlib import Path
_threads = str(max(1, min(int(os.environ.get('BHI_THREADS', '48')), 96)))
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, _threads)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
MATERIALS = Path('/workspace/pulse-training/runs/BHI_router_foundation_compare_20261029_r01/materials')
GROUPS = Path('/workspace/pulse-training/runs/BHI_feature_group_design_20261101/FEATURE_GROUPS.json')
ROOT = Path(os.environ.get('BHI_ROOT', '/workspace/pulse-training/runs/BHI_structured_router_20261101'))
TARGETS = ('SBP', 'DBP')

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(int(os.environ.get('BHI_THREADS', '48')), 96))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

def preprocess_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    std = np.nanstd(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    scale = (q75 - q25) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-08), scale, np.where(np.isfinite(std) & (std > 1e-08), std, 1.0))
    return (median.astype(np.float32), scale.astype(np.float32))

def preprocess_transform(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(x, dtype=np.float32)
    mask = np.isfinite(raw).astype(np.float32)
    values = np.where(mask.astype(bool), raw, median)
    values = np.clip((values - median) / scale, -8.0, 8.0)
    return (values.astype(np.float32), mask)

def record_weights(groups: np.ndarray, bands: np.ndarray, duplicate: np.ndarray) -> np.ndarray:
    unique_groups = np.unique(groups)
    group_bands = []
    for group in unique_groups:
        values = bands[groups == group]
        group_bands.append(int(np.bincount(values, minlength=3).argmax()))
    group_bands = np.asarray(group_bands, dtype=np.int64)
    group_counts = np.bincount(group_bands, minlength=3).astype(np.float64)
    class_power = float(os.environ.get('BHI_CLASS_POWER', '1.0'))
    group_weight = np.power(np.maximum(group_counts, 1.0), -class_power)[group_bands]
    group_weight_by_name = dict(zip(unique_groups.tolist(), group_weight))
    output = np.zeros(len(groups), dtype=np.float64)
    for group in unique_groups:
        mask = groups == group
        local_duplicate = np.asarray(duplicate[mask], dtype=np.float64)
        total = local_duplicate.sum()
        output[mask] = (local_duplicate / total if total > 0 else 1.0 / mask.sum()) * group_weight_by_name[group]
    return output / output.mean()

def expand_group_batch(selected_groups: list[str], group_to_indices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    index_parts = []
    group_parts = []
    for group_id, group in enumerate(selected_groups):
        indices = group_to_indices[group]
        index_parts.append(indices)
        group_parts.append(np.full(len(indices), group_id, dtype=np.int64))
    return (np.concatenate(index_parts), np.concatenate(group_parts))

class FlatRouter(nn.Module):

    def __init__(self, raw_dim: int, relation_dim: int, width: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(raw_dim * 2 + relation_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 3))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((values, mask, relation), dim=-1))

class SemanticBlockRouter(nn.Module):

    def __init__(self, block_indices: list[np.ndarray], relation_dim: int, width: int, dropout: float, heads: int, head_dim: int=3) -> None:
        super().__init__()
        self.blocks = [torch.as_tensor(index) for index in block_indices]
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(2 * len(index), width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout)) for index in block_indices])
        self.relation_encoder = nn.Sequential(nn.Linear(relation_dim * 2, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout))
        self.block_embedding = nn.Parameter(torch.zeros(len(block_indices) + 1, width))
        nn.init.normal_(self.block_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, head_dim))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        tokens = []
        for index, encoder in zip(self.blocks, self.encoders):
            index = index.to(values.device)
            block = torch.cat((values[:, index], mask[:, index]), dim=-1)
            tokens.append(encoder(block))
        relation_mask = torch.ones_like(relation)
        tokens.append(self.relation_encoder(torch.cat((relation, relation_mask), dim=-1)))
        sequence = torch.stack(tokens, dim=1)
        sequence = sequence + self.block_embedding[None]
        sequence = self.fusion(sequence)
        pooled = torch.cat((sequence.mean(dim=1), sequence.max(dim=1).values), dim=-1)
        return self.head(pooled)

class MetricFamilyRouter(nn.Module):

    def __init__(self, family_indices: np.ndarray, context_indices: list[np.ndarray], relation_dim: int, width: int, dropout: float, heads: int, head_dim: int=3) -> None:
        super().__init__()
        self.family_indices = torch.as_tensor(family_indices)
        self.same_site = nn.Conv1d(2, width, kernel_size=3, padding=1, dilation=1)
        self.same_wavelength = nn.Conv1d(2, width, kernel_size=3, padding=4, dilation=4)
        self.family_projection = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
        self.family_embedding = nn.Parameter(torch.zeros(family_indices.shape[0], width))
        nn.init.normal_(self.family_embedding, std=0.02)
        self.context_encoders = nn.ModuleList([nn.Sequential(nn.Linear(2 * len(index), width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout)) for index in context_indices])
        self.context_indices = [torch.as_tensor(index) for index in context_indices]
        self.relation_encoder = nn.Sequential(nn.Linear(relation_dim * 2, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout))
        token_count = family_indices.shape[0] + len(context_indices) + 1
        self.context_embedding = nn.Parameter(torch.zeros(token_count, width))
        nn.init.normal_(self.context_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width * 2, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, head_dim))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        family_indices = self.family_indices.to(values.device)
        family_values = values[:, family_indices]
        family_mask = mask[:, family_indices]
        batch_size, family_count, _ = family_values.shape
        channels = torch.stack((family_values.reshape(batch_size * family_count, 16), family_mask.reshape(batch_size * family_count, 16)), dim=1)
        encoded = self.same_site(channels) + self.same_wavelength(channels)
        encoded = F.gelu(encoded)
        encoded_mask = family_mask.reshape(batch_size * family_count, 1, 16)
        encoded = encoded * encoded_mask
        counts = encoded_mask.sum(dim=-1).clamp_min(1.0)
        means = encoded.sum(dim=-1) / counts
        maxima = encoded.masked_fill(encoded_mask == 0, torch.finfo(encoded.dtype).min).max(dim=-1).values
        maxima = torch.where(encoded_mask.sum(dim=-1) > 0, maxima, torch.zeros_like(maxima))
        family_tokens = self.family_projection(torch.cat((means, maxima), dim=-1)).reshape(batch_size, family_count, -1)
        family_tokens = family_tokens + self.family_embedding[None]
        tokens = [family_tokens]
        for index, encoder in zip(self.context_indices, self.context_encoders):
            index = index.to(values.device)
            block = torch.cat((values[:, index], mask[:, index]), dim=-1)
            tokens.append(encoder(block)[:, None])
        relation_mask = torch.ones_like(relation)
        tokens.append(self.relation_encoder(torch.cat((relation, relation_mask), dim=-1))[:, None])
        sequence = torch.cat(tokens, dim=1)
        sequence = sequence + self.context_embedding[None]
        sequence = self.fusion(sequence)
        pooled = torch.cat((sequence.mean(dim=1), sequence.max(dim=1).values), dim=-1)
        return self.head(pooled)
CHANNEL_ORDER = ['IR_1', 'R_1', 'G_1', 'B_1', 'IR_2', 'R_2', 'G_2', 'B_2', 'IR_3', 'R_3', 'G_3', 'B_3', 'IR_4', 'R_4', 'G_4', 'B_4']

def load_metric_structure() -> tuple[np.ndarray, list[np.ndarray]]:
    payload = json.loads(GROUPS.read_text())
    feature_names = json.loads(Path('/workspace/pulse-training/runs/BHI_unified_bias_20261013_v3/data/new_feature_names.json').read_text())
    name_to_feature = {name: index for index, name in enumerate(feature_names)}
    family_rows = []
    for family in payload['exact_family_tokens']:
        by_channel = {}
        for name in family['names']:
            match = re.match('^(?:supported_)?PPG_(IR|R|G|B)_([1-4])_', name)
            if match is None:
                raise AssertionError(name)
            by_channel[f'{match.group(1)}_{match.group(2)}'] = name_to_feature[name]
        if set(by_channel) != set(CHANNEL_ORDER):
            raise AssertionError(family['family'])
        family_rows.append([by_channel[channel] for channel in CHANNEL_ORDER])
    family_indices = np.asarray(family_rows, dtype=np.int64)
    assert family_indices.shape == (77, 16)
    family_flat = family_indices.reshape(-1)
    assert len(np.unique(family_flat)) == len(family_flat)
    index_map = payload['group_indices']
    context_groups = [['static_demographic'], ['ecg_rhythm', 'session_quality', 'channel_selection', 'supported_global'], ['channel_normalizer', 'cross_channel', 'channel_aggregate'], ['ppg_core_category']]
    contexts = [np.asarray([index for name in names for index in index_map[name]], dtype=np.int64) for names in context_groups]
    context_flat = np.concatenate(contexts)
    assert len(np.unique(context_flat)) == len(context_flat)
    assert np.array_equal(np.sort(np.concatenate((family_flat, context_flat))), np.arange(1543))
    return (family_indices, contexts)

def make_model(variant: str, raw_dim: int, relation_dim: int, block_indices: list[np.ndarray]) -> nn.Module:
    width = int(os.environ.get('BHI_WIDTH', '48'))
    dropout = float(os.environ.get('BHI_DROPOUT', '0.12'))
    if variant == 'flat':
        return FlatRouter(raw_dim, relation_dim, width, dropout)
    if variant == 'blocks':
        return SemanticBlockRouter(block_indices=block_indices, relation_dim=relation_dim, width=width, dropout=dropout, heads=4)
    if variant not in ('metric_blocks', 'bag_metric', 'hier_metric'):
        raise ValueError(variant)
    family_indices, context_indices = load_metric_structure()
    return MetricFamilyRouter(family_indices=family_indices, context_indices=context_indices, relation_dim=relation_dim, width=width, dropout=dropout, heads=4, head_dim=2 if variant == 'hier_metric' else 3)

def class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 3:
        return torch.softmax(logits, dim=-1)
    normal = torch.sigmoid(logits[:, 0])
    high_given_abnormal = torch.sigmoid(logits[:, 1])
    abnormal = 1.0 - normal
    return torch.stack((abnormal * (1.0 - high_given_abnormal), normal, abnormal * high_given_abnormal), dim=-1)

def evaluate(model: nn.Module, values: np.ndarray, mask: np.ndarray, relation: np.ndarray, truth: np.ndarray) -> dict:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(relation))
        probability = class_probabilities(logits).numpy()
    choice = probability.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for expected, selected in zip(truth, choice):
        matrix[int(expected), int(selected)] += 1
    rows = matrix.sum(axis=1)
    recall = np.divide(np.diag(matrix), rows, out=np.zeros(3), where=rows > 0)
    return {'matrix': matrix.tolist(), 'row_totals': rows.tolist(), 'recall': recall.tolist(), 'minimum_recall': float(recall.min()), 'macro_recall': float(recall.mean()), 'accuracy': float(np.trace(matrix) / matrix.sum()), 'probability': probability}

def train_model(*, variant: str, values: np.ndarray, mask: np.ndarray, relation: np.ndarray, band: np.ndarray, groups: np.ndarray, duplicate: np.ndarray, block_indices: list[np.ndarray], schedule_epochs: int, max_epochs: int, seed: int, val_values: np.ndarray | None, val_mask: np.ndarray | None, val_relation: np.ndarray | None, val_band: np.ndarray | None) -> tuple[nn.Module, int, list[dict]]:
    set_seed(seed)
    model = make_model(variant, values.shape[1], relation.shape[1], block_indices)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get('BHI_LR', '8e-4')), weight_decay=float(os.environ.get('BHI_WEIGHT_DECAY', '1e-3')))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(schedule_epochs, 1))
    unique_groups = np.unique(groups)
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    value_tensor = torch.from_numpy(values)
    mask_tensor = torch.from_numpy(mask)
    relation_tensor = torch.from_numpy(relation)
    band_tensor = torch.from_numpy(band)
    record_weight_tensor = torch.from_numpy(record_weights(groups, band, duplicate))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_score = (-float('inf'), -float('inf'), -float('inf'))
    patience = int(os.environ.get('BHI_PATIENCE', '18'))
    history = []
    batch_groups = int(os.environ.get('BHI_BATCH_GROUPS', '64'))
    consistency_weight = float(os.environ.get('BHI_GROUP_CONSISTENCY_WEIGHT', '0.25'))
    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = rng.permutation(unique_groups)
        losses = []
        for start in range(0, len(shuffled), batch_groups):
            selected = shuffled[start:start + batch_groups].tolist()
            indices, group_ids = expand_group_batch(selected, group_to_indices)
            index_tensor = torch.from_numpy(indices)
            group_tensor = torch.from_numpy(group_ids)
            logits = model(value_tensor[index_tensor], mask_tensor[index_tensor], relation_tensor[index_tensor])
            weights = record_weight_tensor[index_tensor]
            if logits.shape[-1] == 3:
                record_losses = F.cross_entropy(logits, band_tensor[index_tensor], reduction='none', label_smoothing=0.02)
                record_loss = (record_losses * weights).sum() / weights.sum()
            else:
                local_band = band_tensor[index_tensor]
                normal_target = (local_band == 1).float()
                normal_per_item = F.binary_cross_entropy_with_logits(logits[:, 0], normal_target, reduction='none')
                normal_loss = (normal_per_item * weights).sum() / weights.sum()
                endpoint_mask = local_band != 1
                endpoint_target = (local_band[endpoint_mask] == 2).float()
                endpoint_weights = weights[endpoint_mask]
                low_count = int((endpoint_target == 0).sum())
                high_count = int((endpoint_target == 1).sum())
                pos_weight = torch.tensor(max(low_count, 1) / max(high_count, 1), dtype=logits.dtype)
                endpoint_per_item = F.binary_cross_entropy_with_logits(logits[endpoint_mask, 1], endpoint_target, reduction='none', pos_weight=pos_weight)
                endpoint_loss = (endpoint_per_item * endpoint_weights).sum() / endpoint_weights.sum()
                record_loss = normal_loss + endpoint_loss
            group_count = len(selected)
            record_probability = class_probabilities(logits)
            group_probability_sum = torch.zeros(group_count, 3, dtype=logits.dtype)
            group_probability_sum.index_add_(0, group_tensor, record_probability)
            counts = torch.zeros(group_count, dtype=logits.dtype)
            counts.index_add_(0, group_tensor, torch.ones(len(logits), dtype=logits.dtype))
            group_probability = group_probability_sum / counts[:, None]
            bag_loss_weight = 0.25 if variant == 'bag_metric' else 0.0
            if bag_loss_weight:
                local_band_numpy = band_tensor[index_tensor].numpy()
                group_band_tensor = torch.tensor([int(np.bincount(local_band_numpy[group_ids == group_id], minlength=3).argmax()) for group_id in range(group_count)], dtype=torch.long)
                group_band_counts = torch.bincount(group_band_tensor, minlength=3).float()
                bag_weights = 3.0 / group_band_counts[group_band_tensor].clamp_min(1.0)
                bag_weights = bag_weights / bag_weights.mean()
                bag_loss = -torch.log(group_probability[torch.arange(group_count), group_band_tensor].clamp_min(1e-08))
                bag_loss = (bag_loss * bag_weights).mean()
            else:
                bag_loss = torch.zeros((), dtype=logits.dtype)
            consistency = F.kl_div(torch.log(record_probability.clamp_min(1e-08)), group_probability.detach()[group_tensor], reduction='batchmean')
            loss = record_loss + consistency_weight * consistency + bag_loss_weight * bag_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses))}
        if val_values is not None:
            metrics = evaluate(model, val_values, val_mask, val_relation, val_band)
            score = (metrics['macro_recall'], metrics['minimum_recall'], metrics['accuracy'])
            row.update({'val_minimum_recall': metrics['minimum_recall'], 'val_macro_recall': metrics['macro_recall'], 'val_accuracy': metrics['accuracy'], 'val_matrix': metrics['matrix']})
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= patience:
                history.append(row)
                break
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    model.load_state_dict(best_state)
    return (model, best_epoch, history)

def choose_policy(probability: np.ndarray, truth: np.ndarray) -> dict:
    low_values = np.unique(np.quantile(probability[:, 0], np.linspace(0.02, 0.98, 49)))
    high_values = np.unique(np.quantile(probability[:, 2], np.linspace(0.02, 0.98, 49)))
    low_values = np.concatenate(([0.0], low_values, [1.01]))
    high_values = np.concatenate(([0.0], high_values, [1.01]))
    best = None
    for low_threshold in low_values:
        for high_threshold in high_values:
            choice = decide(probability, float(low_threshold), float(high_threshold))
            matrix = np.zeros((3, 3), dtype=np.int64)
            for expected, selected in zip(truth, choice):
                matrix[int(expected), int(selected)] += 1
            rows = matrix.sum(axis=1)
            recall = np.divide(np.diag(matrix), rows, out=np.zeros(3), where=rows > 0)
            score = (float(recall.min()), float(recall.mean()), float(np.trace(matrix) / matrix.sum()))
            candidate = (score, float(low_threshold), float(high_threshold))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return {'low_threshold': best[1], 'high_threshold': best[2], 'objective': list(best[0])}

def decide(probability: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    low = probability[:, 0]
    high = probability[:, 2]
    choice = np.ones(len(probability), dtype=np.int64)
    choice[(low >= low_threshold) & (low >= high)] = 0
    choice[(high >= high_threshold) & (high > low)] = 2
    return choice

def confusion_metrics(truth: np.ndarray, choice: np.ndarray, groups: np.ndarray) -> dict:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for expected, selected in zip(truth, choice):
        matrix[int(expected), int(selected)] += 1
    rows = matrix.sum(axis=1)
    columns = matrix.sum(axis=0)
    recall = np.divide(np.diag(matrix), rows, out=np.zeros(3), where=rows > 0)
    precision = np.divide(np.diag(matrix), columns, out=np.zeros(3), where=columns > 0)
    return {'matrix': matrix.tolist(), 'row_totals': rows.tolist(), 'column_totals': columns.tolist(), 'recall': recall.tolist(), 'precision': precision.tolist(), 'accuracy': float(np.trace(matrix) / matrix.sum()), 'macro_recall': float(recall.mean()), 'minimum_recall': float(recall.min()), 'records': int(len(truth)), 'groups': int(len(np.unique(groups)))}

def load_groups() -> tuple[list[str], list[np.ndarray]]:
    payload = json.loads(GROUPS.read_text())
    preferred = ['static_demographic', 'ecg_rhythm', 'session_quality', 'channel_selection', 'supported_global', 'ppg_core_quality', 'ppg_core_timing', 'ppg_core_morphology', 'ppg_core_spectral', 'ppg_core_category', 'ppg_supported_quality', 'ppg_supported_waveform', 'ppg_supported_pat', 'channel_normalizer', 'cross_channel', 'channel_aggregate']
    index_map = payload['group_indices']
    assert set(preferred) == set(index_map)
    indices = [np.asarray(index_map[name], dtype=np.int64) for name in preferred]
    covered = np.concatenate(indices)
    assert len(covered) == 1543
    assert np.array_equal(np.sort(covered), np.arange(1543))
    return (preferred, indices)

def split_role(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(x[:, :1543], dtype=np.float32)
    relation = np.asarray(x[:, 1543:], dtype=np.float32)
    assert relation.shape[1] == 20
    assert np.isfinite(relation).all()
    return (raw, relation)

def run_target(target: str, variant: str) -> None:
    seed_offset = int(os.environ.get('BHI_SEED_OFFSET', '0'))
    output = ROOT / variant / target
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    group_names, block_indices = load_groups()
    target_index = TARGETS.index(target)
    truth_parts = []
    choice_parts = []
    group_parts = []
    folds = []
    started = time.time()
    for outer in range(3):
        material = MATERIALS / f'fold_{outer}'
        with np.load(material / 'train.npz', allow_pickle=False) as source:
            train_raw, train_relation = split_role(source['X'])
            train_band = source['y'][:, target_index].astype(np.int64)
            train_groups = source['groups']
            train_duplicate = source['duplicate_weight']
        with np.load(material / 'val.npz', allow_pickle=False) as source:
            val_raw, val_relation = split_role(source['X'])
            val_band = source['y'][:, target_index].astype(np.int64)
            val_groups = source['groups']
        with np.load(material / 'fit_all.npz', allow_pickle=False) as source:
            fit_raw, fit_relation = split_role(source['X'])
            fit_band = source['y'][:, target_index].astype(np.int64)
            fit_groups = source['groups']
            fit_duplicate = source['duplicate_weight']
        with np.load(material / 'query.npz', allow_pickle=False) as source:
            query_raw, query_relation = split_role(source['X'])
            query_band = source['y'][:, target_index].astype(np.int64)
            query_groups = source['groups']
            query_indices = source['indices']
        if set(train_groups) & set(val_groups):
            raise AssertionError('inner group leakage')
        if set(fit_groups) & set(query_groups):
            raise AssertionError('outer group leakage')
        median, scale = preprocess_fit(train_raw)
        train_values, train_mask = preprocess_transform(train_raw, median, scale)
        val_values, val_mask = preprocess_transform(val_raw, median, scale)
        selection_model, selected_epoch, history = train_model(variant=variant, values=train_values, mask=train_mask, relation=train_relation, band=train_band, groups=train_groups, duplicate=train_duplicate, block_indices=block_indices, schedule_epochs=int(os.environ.get('BHI_EPOCHS', '60')), max_epochs=int(os.environ.get('BHI_EPOCHS', '60')), seed=20261101 + seed_offset + outer * 100 + target_index, val_values=val_values, val_mask=val_mask, val_relation=val_relation, val_band=val_band)
        val_metrics = evaluate(selection_model, val_values, val_mask, val_relation, val_band)
        policy = choose_policy(val_metrics['probability'], val_band)
        del selection_model
        fit_median, fit_scale = preprocess_fit(fit_raw)
        fit_values, fit_mask = preprocess_transform(fit_raw, fit_median, fit_scale)
        query_values, query_mask = preprocess_transform(query_raw, fit_median, fit_scale)
        final_model, _, final_history = train_model(variant=variant, values=fit_values, mask=fit_mask, relation=fit_relation, band=fit_band, groups=fit_groups, duplicate=fit_duplicate, block_indices=block_indices, schedule_epochs=int(os.environ.get('BHI_EPOCHS', '60')), max_epochs=selected_epoch, seed=20261101 + seed_offset + outer * 100 + target_index + 50, val_values=None, val_mask=None, val_relation=None, val_band=None)
        query_probability = evaluate(final_model, query_values, query_mask, query_relation, query_band)['probability']
        query_choice = decide(query_probability, policy['low_threshold'], policy['high_threshold'])
        query_metrics = confusion_metrics(query_band, query_choice, query_groups)
        torch.save({'state_dict': final_model.state_dict(), 'target': target, 'variant': variant, 'selected_epoch': selected_epoch, 'block_names': group_names}, output / f'fold_{outer}.pt')
        np.savez_compressed(output / f'fold_{outer}_predictions.npz', query_indices=query_indices, query_groups=query_groups, true_band=query_band, probability=query_probability, choice=query_choice)
        folds.append({'outer': outer, 'selected_epoch': selected_epoch, 'parameter_count': sum((parameter.numel() for parameter in final_model.parameters())), 'policy': policy, 'query': query_metrics, 'history': history, 'final_history': final_history[-1:]})
        truth_parts.append(query_band)
        choice_parts.append(query_choice)
        group_parts.append(query_groups)
    truth = np.concatenate(truth_parts)
    choice = np.concatenate(choice_parts)
    groups = np.concatenate(group_parts)
    report = {'schema': 'BHI_STRUCTURED_ROUTER_SCREEN_V1', 'status': 'PASS', 'target': target, 'variant': variant, 'class_power': float(os.environ.get('BHI_CLASS_POWER', '1.0')), 'block_names': group_names, 'records': int(len(truth)), 'groups': int(len(np.unique(groups))), 'record_inference': 'single record only', 'query_labels_used_for_policy': False, 'aggregate': confusion_metrics(truth, choice, groups), 'folds': folds, 'elapsed_seconds': time.time() - started}
    write_json(output / 'RESULTS.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--variant', choices=('flat', 'blocks', 'metric_blocks', 'bag_metric', 'hier_metric'), required=True)
    args = parser.parse_args()
    if os.getuid() != 1003:
        raise AssertionError('Run as uid 1003')
    ROOT.mkdir(parents=True, exist_ok=True)
    run_target(args.target, args.variant)
if __name__ == '__main__':
    main()
