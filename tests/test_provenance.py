"""Regression checks for metadata binding and prediction-cache input identity."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cache
sys.path.insert(0,str(ROOT/'src/probe_cmp'))
import run

with tempfile.TemporaryDirectory(prefix='e0-provenance-test-') as tmp:
    d=Path(tmp);meta=d/'meta.parquet';acts=d/'acts.npy'
    t=pd.DataFrame({'trace_id':['a','a'],'row':[0,1],'pos':[128,256],'prefix':['x','xy']})
    t.to_parquet(meta);np.save(acts,np.ones((2,1,3)))
    cfg={'layers':[20]};mh=cache.write_dataset_manifest(str(meta),str(acts),cfg,cache.sha_obj(cfg),t.trace_id.tolist(),(2,1,3))
    cache.check_dataset_manifest(str(meta),str(acts),t.trace_id.tolist(),(2,1,3),strict=True)
    swapped=t.copy();swapped[['pos','prefix']]=t[['pos','prefix']].iloc[::-1].to_numpy();swapped.to_parquet(meta)
    for strict in [False,True]:
        try:
            cache.check_dataset_manifest(str(meta),str(acts),t.trace_id.tolist(),(2,1,3),strict=strict)
        except cache.StaleCache as exc:assert 'metadata contents changed' in str(exc)
        else:raise AssertionError('swapped metadata accepted')
    t.to_parquet(meta)
    mp=Path(cache.dataset_manifest_path(str(meta)));original=mp.read_text()
    legacy=json.loads(original);legacy.pop('meta_sha256');mp.write_text(json.dumps(legacy))
    try:cache.check_dataset_manifest(str(meta),str(acts),t.trace_id.tolist(),(2,1,3))
    except cache.StaleCache as exc:assert 'legacy manifest' in str(exc)
    else:raise AssertionError('unbound legacy metadata accepted')
    no_hash=json.loads(original);no_hash['acts_sha256']=None;mp.write_text(json.dumps(no_hash))
    try:cache.check_dataset_manifest(str(meta),str(acts),t.trace_id.tolist(),(2,1,3),strict=True)
    except cache.StaleCache as exc:assert 'requires an activation hash' in str(exc)
    else:raise AssertionError('strict accepted missing activation hash')

    pop=pd.DataFrame({'trace_id':['a','b','c','d'],'problem_id':[1,1,2,2],'y':[0,1,0,1]})
    p0=run.population_sha(pop,['4096']);changed=pop.copy();changed.y=1-changed.y
    assert p0!=run.population_sha(changed,['4096'])
    changed=pop.copy();changed.problem_id=[3,3,4,4]
    assert p0!=run.population_sha(changed,['4096'])
    (d/'traces.jsonl').write_text('{}\n')
    src={'meta':'meta.parquet','acts':'acts.npy'}
    store=SimpleNamespace(reg={'traces':'traces.jsonl','sources':[src]},sources={'full':(src,t)})
    fp0=run.input_fingerprint(store,str(d))
    assert fp0==run.input_fingerprint(store,str(d))
    np.save(acts,np.zeros((2,1,3)))
    fp1=run.input_fingerprint(store,str(d));assert fp0!=fp1
    swapped.to_parquet(meta);fp2=run.input_fingerprint(store,str(d));assert fp1!=fp2
    (d/'traces.jsonl').write_text('{"changed":true}\n')
    assert fp2!=run.input_fingerprint(store,str(d))
    assert 'run.py' in fp2['code'] and 'models.py' in fp2['code']
    old=run.sha({'population_sha':p0,'inputs':fp0,'folds':{'0':{'1':0}}})
    new=run.sha({'population_sha':p0,'inputs':fp1,'folds':{'0':{'1':0}}})
    pred=d/'pred.parquet';pd.DataFrame({'cfg_sha':[old],'p':[.5]}).to_parquet(pred)
    assert run.saved_ok(pred,old) and not run.saved_ok(pred,new)
    assert old!=run.sha({'population_sha':p0,'inputs':fp0,'folds':{'0':{'1':1}}})
print('All provenance regression checks passed.')
