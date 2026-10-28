#!/usr/bin/env python3
"""Censored Weibull AFT baseline on the frozen E0 exact-only source folds.

No graph, notch, ambiguous, retest or synthetic records enter this script.
The model is intentionally stress-only and fixed before inspecting test scores.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize


LN10 = math.log(10)


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(data[0]))
        wr.writeheader()
        wr.writerows(data)


def loss_grad(params, stress, z, event):
    """Negative observed-data log likelihood for Z=log10(N)."""
    alpha, beta, log_k = params
    k = np.exp(log_k)
    x = np.log2(stress / 100.0)
    mu = alpha + beta*x
    h = LN10*k*(z-mu)
    # A wide search box can enter a numerically irrelevant exp-overflow region.
    # Linear continuation beyond h=60 preserves a useful outward derivative.
    t = np.exp(np.minimum(h, 60.0))
    high = h > 60.0
    if np.any(high):
        t[high] *= 1.0 + h[high]-60.0
    loss = np.sum(t - event*(log_k+math.log(LN10)+h))
    dt_dh = np.where(high, math.exp(60.0), t)
    dloss_dh = dt_dh-event
    g_alpha = -LN10*k*np.sum(dloss_dh)
    g_beta = -LN10*k*np.dot(dloss_dh,x)
    g_logk = np.dot(dloss_dh,h)-np.sum(event)
    return float(loss), np.array([g_alpha,g_beta,g_logk],dtype=float)


def fit(train):
    stress = np.array([float(r['stress_amplitude_MPa']) for r in train])
    z = np.log10(np.array([float(r['cycles_failure_or_bound']) for r in train]))
    event = np.array([float(r['failure_event_1']) for r in train])
    x = np.log2(stress/100)
    ols = np.polyfit(x[event==1],z[event==1],1)
    beta0 = float(np.clip(ols[0],-7.0,-0.05))
    alpha0 = float(np.clip(np.median(z[event==1])-beta0*np.median(x[event==1]),1,11))
    starts = [np.array([alpha0,beta0,0.0]),
              np.array([alpha0,-1.0,-0.5]),
              np.array([alpha0,-3.0,0.5])]
    bounds = [(0.0,12.0),(-12.0,0.0),(-4.0,3.0)]
    solutions = [minimize(loss_grad,start,args=(stress,z,event),method='L-BFGS-B',
                          jac=True,bounds=bounds,options={'maxiter':3000,'ftol':1e-12})
                 for start in starts]
    valid = [r for r in solutions if r.success and np.isfinite(r.fun)]
    if not valid:
        raise RuntimeError('No converged Weibull likelihood fit')
    best = min(valid,key=lambda r:r.fun)
    return best, stress


def score_one(params, stress, cycles, event):
    alpha,beta,log_k = params
    k = math.exp(log_k)
    mu = alpha+beta*math.log2(stress/100)
    z = math.log10(cycles)
    h = LN10*k*(z-mu)
    # If the tail exceeds floating-point range, retain its log10 magnitude.
    t = math.exp(h) if h<700 else math.inf
    ll = (log_k+math.log(LN10)+h if event else 0.0)-t
    median_z = mu+math.log10(math.log(2))/k
    return mu,median_z,-ll,(-ll if event else 0.0),(t if not event else 0.0),h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('.'),help='Repository root')
    ap.add_argument('--output',type=Path,default=None)
    a = ap.parse_args()
    path = a.root/'data/cohorts/smooth_core_66.csv'
    manifest_path = a.root/'data/validation/record_membership_2026-10-02.csv'
    if not path.exists():
        path = a.root/'no_author_dependency/smooth_core_66.csv'
        manifest_path = a.root/'validation/record_membership_2026-10-02.csv'
    hash_path = manifest_path.parent/'input_hashes_2026-10-02.json'
    if hash_path.exists():
        expected = json.loads(hash_path.read_text(encoding='utf-8'))['input_sha256']['smooth']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Smooth-core input differs from the frozen split source')
    source = rows(path)
    manifest = [r for r in rows(manifest_path) if r['analysis']=='primary_exact_LOSO']
    assert len(source)==66 and len(manifest)==198
    assert all(r['failure_event_1'] in ('0','1') and float(r['stress_amplitude_MPa'])>0
               for r in source)
    by_id = {r['record_id']:r for r in source}
    assert len(by_id)==66
    output = a.output or a.root/'results/exact_weibull_baseline'
    output.mkdir(parents=True,exist_ok=True)
    preds=[];summaries=[];coeffs=[]
    for fold_id in ['E0_Wu_2021','E0_Romano_2018','E0_Chen_2024']:
        m = [r for r in manifest if r['fold_id']==fold_id]
        train = [by_id[r['record_key'].removeprefix('EXACT:')] for r in m if r['role']=='train']
        test = [by_id[r['record_key'].removeprefix('EXACT:')] for r in m if r['role']=='test']
        assert len(train)+len(test)==66
        assert {r['record_id'] for r in train+test}==set(by_id)
        assert not ({r['study_id'] for r in train}&{r['study_id'] for r in test})
        assert {r['study_id'] for r in test}=={fold_id.removeprefix('E0_')}
        sol, train_stress = fit(train)
        lo,hi=float(train_stress.min()),float(train_stress.max())
        train_R={r['stress_ratio_R'] for r in train}
        alpha,beta,logk=map(float,sol.x)
        coeffs.append(dict(fold_id=fold_id,held_out_study=test[0]['study_id'],
                           train_n=len(train),train_fail=sum(int(r['failure_event_1']) for r in train),
                           train_runout=sum(1-int(r['failure_event_1']) for r in train),
                           alpha_log10_scale_at_100MPa=alpha,
                           beta_log10_scale_per_stress_doubling=beta,
                           weibull_shape_k=math.exp(logk),training_negative_log_likelihood=float(sol.fun),
                           optimizer_success=bool(sol.success),optimizer_message=str(sol.message),
                           beta_at_constraint=int(beta<=-11.999999 or beta>=-0.000001),
                           k_at_constraint=int(logk<=-3.999999 or logk>=2.999999)))
        fold_preds=[]
        for r in test:
            n=float(r['cycles_failure_or_bound']);ev=int(r['failure_event_1'])
            stress=float(r['stress_amplitude_MPa'])
            mu,median_z,nll,fail_term,censor_term,h=score_one(sol.x,stress,n,ev)
            pred=dict(fold_id=fold_id,record_id=r['record_id'],study_id=r['study_id'],
                      event_failure=ev,cycles_or_bound=int(n),bound_operator=r['bound_operator'],
                      stress_amplitude_MPa=stress,stress_ratio_R=float(r['stress_ratio_R']),
                      train_stress_min_MPa=lo,train_stress_max_MPa=hi,
                      stress_outside_training_range=int(stress<lo or stress>hi),
                      stress_ratio_unseen=int(r['stress_ratio_R'] not in train_R),
                      predicted_log10_weibull_scale=mu,predicted_log10_median_life=median_z,
                      observed_log10_failure_life=math.log10(n) if ev else '',
                      censored_negative_log_likelihood=nll,
                      failure_density_NLL_component=fail_term if ev else '',
                      runout_survival_NLL_component=censor_term if not ev else '')
            preds.append(pred);fold_preds.append(pred)
        fails=[r for r in fold_preds if r['event_failure']==1]
        finite=all(math.isfinite(r['censored_negative_log_likelihood']) for r in fold_preds)
        summaries.append(dict(fold_id=fold_id,held_out_study=test[0]['study_id'],
                              test_n=len(test),test_fail=len(fails),test_runout=len(test)-len(fails),
                              mean_censored_NLL_log10_density=(sum(r['censored_negative_log_likelihood'] for r in fold_preds)/len(fold_preds)) if finite else '',
                              failure_only_MAE_log10_cycles=sum(abs(r['predicted_log10_median_life']-r['observed_log10_failure_life']) for r in fails)/len(fails),
                              total_failure_density_NLL=sum(r['failure_density_NLL_component'] for r in fails),
                              total_runout_survival_NLL=sum(r['runout_survival_NLL_component'] for r in fold_preds if not r['event_failure']),
                              stress_outside_training_range_n=sum(r['stress_outside_training_range'] for r in fold_preds),
                              stress_ratio_unseen_n=sum(r['stress_ratio_unseen'] for r in fold_preds),
                              nonfinite_score_n=sum(not math.isfinite(r['censored_negative_log_likelihood']) for r in fold_preds),
                              score_status='finite' if finite else 'nonfinite_extrapolation'))
    write_rows(output/'heldout_predictions_E0_66.csv',preds)
    write_rows(output/'fold_metrics_E0_3.csv',summaries)
    write_rows(output/'fit_parameters_E0_3.csv',coeffs)
    meta={'input_sha256':{str(path.relative_to(a.root)):hashlib.sha256(path.read_bytes()).hexdigest(),
                          str(manifest_path.relative_to(a.root)):hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
          'software_versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
          'model':'Weibull AFT on cycles; log10 scale alpha+beta*log2(stress_amplitude_MPa/100); k shared per fit; beta<=0',
          'predictors':['stress_amplitude_MPa'],
          'likelihood':'failures log density of Z=log10 cycles; runouts log survival above bound',
          'fixed_bounds':{'alpha':[0,12],'beta':[-12,0],'log_shape':[-4,3]},
          'folds':[r['fold_id'] for r in summaries],
          'reporting':'no graph data or synthetic data; no record-level random split'}
    (output/'run_metadata_E0.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'folds':summaries,'parameters':coeffs,'output':str(output)},allow_nan=False))


if __name__=='__main__':
    main()
