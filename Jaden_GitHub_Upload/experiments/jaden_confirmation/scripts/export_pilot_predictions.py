"""Convert one saved probe_cmp checkpoint/seed comparison into planning CSV.

This reads existing OOF predictions; it does not fit a model or certify their provenance.
The source pipeline stores y=1 and p for WRONG. Output consistently means CORRECT.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def convert(frame, checkpoint, seed, probe_model, probe_feature, probe_layer,
            baseline_model, baseline_feature, baseline_layer):
    required = {'ckpt', 'seed', 'model', 'feature', 'layer', 'trace_id',
                'problem_id', 'y', 'p', 'cfg_sha'}
    if not required <= set(frame.columns):
        raise ValueError('Missing columns: ' + str(sorted(required - set(frame.columns))))
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing values in saved prediction columns')
    chosen = frame[(frame.ckpt.astype(str) == str(checkpoint)) & (frame.seed == seed)].copy()
    if chosen.empty:
        raise ValueError('No rows for requested checkpoint and seed; inspect available combinations first')
    if chosen.cfg_sha.nunique() != 1:
        raise ValueError('Mixed configuration hashes; use one validated run')
    def arm(model, feature, layer, name):
        a = chosen[(chosen.model == model) & (chosen.feature == feature) & (chosen.layer == layer)].copy()
        if a.empty or a.trace_id.duplicated().any():
            raise ValueError(f'{name}: empty arm or duplicate trace IDs')
        if not a.y.isin([0, 1]).all() or not np.isfinite(a.p).all() or not a.p.between(0, 1).all():
            raise ValueError(f'{name}: expected binary y and finite P(wrong) in [0,1]')
        return a.set_index('trace_id')[['problem_id', 'y', 'p']].sort_index()
    p = arm(probe_model, probe_feature, probe_layer, 'probe')
    b = arm(baseline_model, baseline_feature, baseline_layer, 'baseline')
    if not p.index.equals(b.index):
        raise ValueError('The arms do not contain identical trace IDs; do not silently intersect them')
    if not p[['problem_id', 'y']].equals(b[['problem_id', 'y']]):
        raise ValueError('Labels or problem IDs differ between arms')
    out = pd.DataFrame({'trace_id': p.index, 'problem_id': p.problem_id.to_numpy(),
                        'checkpoint': str(checkpoint), 'y_correct': 1 - p.y.to_numpy(),
                        'probe_score': 1 - p.p.to_numpy(), 'baseline_score': 1 - b.p.to_numpy()})
    return out, {'cfg_sha': str(chosen.cfg_sha.iloc[0]), 'seed': int(seed),
                 'checkpoint': str(checkpoint), 'rows': len(out),
                 'orientation_conversion': '1-y_wrong; 1-p_wrong',
                 'probe': [probe_model, probe_feature, probe_layer],
                 'baseline': [baseline_model, baseline_feature, baseline_layer],
                 'caution': 'Verify grouped OOF provenance and population selection separately; this converter does not certify them.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--probe-model', required=True)
    ap.add_argument('--probe-feature', required=True)
    ap.add_argument('--probe-layer', type=int, required=True)
    ap.add_argument('--baseline-model', required=True)
    ap.add_argument('--baseline-feature', required=True)
    ap.add_argument('--baseline-layer', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = vars(ap.parse_args())
    source = Path(args.pop('input')); dest = Path(args.pop('out'))
    if dest.exists() or dest.with_suffix('.manifest.json').exists():
        raise ValueError('Output exists; choose a new filename')
    frame = pd.read_parquet(source) if source.suffix == '.parquet' else pd.read_csv(source)
    out, report = convert(frame, **args)
    report['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    dest.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
