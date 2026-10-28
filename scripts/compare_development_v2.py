#!/usr/bin/env python3
"""Frozen v2 exact-only and train-only synthetic comparison.

Run from the repository with:
 python scripts/compare_development_v2.py --root . --out results/development_v2
Requires the existing fit_weibull_exact_baseline.py and fit_m1_source_aware.py.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp

from fit_weibull_exact_baseline import fit as fit_e0
from fit_m1_source_aware import (CHECK_NODES, FIT_NODES, LN10, e0_logterms,
    median_e0, median_m1, prediction_checks, quadrature, study_marginal_objective,
    groups_for)

COHORT = "data/cohorts/development_v2_199_2026-10-06.csv"
FOLDS = "data/validation/development_v2_grouped_fold_assignments_2026-10-06.csv"
FEASIBILITY = "data/validation/development_v2_grouped_fold_feasibility_2026-10-06.csv"
BLOBS = {COHORT:"ee9f6451e60b290c89c44f681b471b3c007c6f74",
         FOLDS:"a4969bc3c6562036a5a980ca248ba78c5359fcba",
         FEASIBILITY:"037337e85e5e7d36cb93a22ec8b4af41e2209e08"}
PRIMARY = ("Wu_2021", "Chen_2024", "Matusu_2026")
EXPECTED = {"Wu_2021":(41,33,8), "Chen_2024":(18,18,0),
            "Matusu_2026":(133,115,18), "Romano_2018":(7,6,1)}
SEEDS=(13,29,47,71,101)
TAU=.25
STOP=10_000_000
CAP=2.

def read(path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))

def write(path, rows):
    if not rows: raise ValueError(f"Empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def blob(path):
    b=path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()

def input_checks(root):
    for relative,expected in BLOBS.items():
        actual=blob(root/relative)
        if actual!=expected: raise ValueError(f"Frozen input differs: {relative} {actual}")
    all_rows=read(root/COHORT);assigned=read(root/FOLDS);feas=read(root/FEASIBILITY)
    if len(all_rows)!=len({r["record_id"] for r in all_rows}):
        raise AssertionError("Invalid cohort uniqueness")
    if len(all_rows)!=199 or len(assigned)!=199 or len(feas)!=7:
        raise AssertionError("Unexpected frozen file dimensions")
    byid={r["record_id"]:r for r in all_rows}
    if set(byid)!={r["record_id"] for r in assigned}: raise AssertionError("Assignment mismatch")
    for r in all_rows:
        a=float(r["stress_amplitude_MPa"]); n=float(r["cycles_failure_or_bound"])
        e=r["failure_event_1"]
        if not (a>0 and n>0 and e in ("0","1")):raise AssertionError(r["record_id"])
        if (e=="1" and r["bound_operator"]!="=") or (e=="0" and r["bound_operator"] not in (">",">=","≥")):
            raise AssertionError(f"Event operator {r['record_id']}")
    for s,(n,nfail,nrun) in EXPECTED.items():
        rows=[r for r in all_rows if r["publication_group_id"]==s]
        if (len(rows),sum(int(r["failure_event_1"]) for r in rows),sum(r["failure_event_1"]=="0" for r in rows))!=(n,nfail,nrun):
            raise AssertionError(f"Count {s}")
    for a in assigned:
        r=byid[a["record_id"]]
        if any(a[k]!=r[k] for k in ("publication_group_id","platform_family_id","stress_ratio_R","failure_event_1")):
            raise AssertionError(f"Assignment metadata {a['record_id']}")
        expected_track=("secondary_Rminus1_transfer" if r["publication_group_id"]=="Romano_2018"
                        else "primary_R01_publication_transfer")
        if a["evaluation_track"]!=expected_track and r["publication_group_id"]!="Romano_2018":
            raise AssertionError(f"Track {a['record_id']}")
        if r["publication_group_id"] in PRIMARY and a["primary_holdout_fold"]!="holdout_"+r["publication_group_id"]:
            raise AssertionError(f"Fold {a['record_id']}")
    primary=[r for r in all_rows if r["publication_group_id"] in PRIMARY]
    if len(primary)!=192 or sum(int(r["failure_event_1"]) for r in primary)!=166:
        raise AssertionError("Primary membership")
    for s in ("Wu_2021","Matusu_2026"):
        if {float(r["cycles_failure_or_bound"]) for r in primary
            if r["publication_group_id"]==s and r["failure_event_1"]=="0"}!={STOP}:
            raise AssertionError(f"Source stop {s}")
    if not all(float(r["stress_ratio_R"])==.1 for r in primary):
        raise AssertionError("Primary R")
    folds={}
    for hold in PRIMARY:
        train=[r for r in primary if r["publication_group_id"]!=hold]
        test=[r for r in primary if r["publication_group_id"]==hold]
        if set(r["publication_group_id"] for r in train)&{hold}:raise AssertionError("Leakage")
        f=next(x for x in feas if x["track"]=="primary_R01_publication_transfer" and x["heldout_group"]==hold)
        counts=(len(train),sum(int(r["failure_event_1"]) for r in train),
                sum(r["failure_event_1"]=="0" for r in train),
                len(test),sum(int(r["failure_event_1"]) for r in test),
                sum(r["failure_event_1"]=="0" for r in test))
        expected=tuple(int(f[k]) for k in ("train_n","train_failures","train_runouts",
                                            "test_n","test_failures","test_runouts"))
        if counts!=expected:raise AssertionError(f"Feasibility {hold} {counts} {expected}")
        folds[hold]=(train,test)
    return folds

def series_key(r):
    s=r["publication_group_id"];rid=r["record_id"]
    if s=="Matusu_2026":
        m=re.match(r"Matusu2026_((?:\d+)|(?:C\d+))_",rid)
        if not m:raise ValueError(f"Unknown workbook series {rid}")
        return m.group(1)
    if s=="Wu_2021":
        if r["build_orientation"] not in ("Horizontal","Vertical"):raise ValueError(f"Unknown Wu orientation {rid}")
        return r["build_orientation"]
    if s=="Chen_2024":
        m=re.match(r"Chen_2024:Chen2024-(L40|L10)-",rid)
        if not m:raise ValueError(f"Unknown Chen group {rid}")
        return m.group(1)
    raise ValueError(s)

def weighted_objective_grad(params, groups, tau, nodes):
    alpha,beta,logk=map(float,params);k=math.exp(logk)
    q,logw=quadrature(nodes);offset=tau*q
    value=0.; gradient=np.zeros(3)
    for rows in groups:
        stress=np.array([float(r["stress_amplitude_MPa"]) for r in rows])
        z=np.log10([float(r["cycles_failure_or_bound"]) for r in rows])
        event=np.array([int(r["failure_event_1"]) for r in rows])
        weights=np.array([float(r.get("weight",1.)) for r in rows])
        x=np.log2(stress/100.)
        h=LN10*k*(z[:,None]-alpha-beta*x[:,None]-offset[None,:])
        if not np.all(np.isfinite(h)) or np.any(h>600):
            return 1e300,np.zeros(3)
        t=np.exp(h)
        ell=event[:,None]*(logk+math.log(LN10)+h)-t
        node=logw+(weights[:,None]*ell).sum(axis=0)
        ls=logsumexp(node)
        if not math.isfinite(ls):return 1e300,np.zeros(3)
        post=np.exp(node-ls)
        da=-LN10*k*(event[:,None]-t)
        db=da*x[:,None]
        dk=event[:,None]*(1+h)-t*h
        grads=np.array([(weights[:,None]*v).sum(axis=0) for v in (da,db,dk)])
        value-=ls;gradient-=grads@post
    return float(value),gradient

def group(rows):
    d=defaultdict(list)
    for r in rows:d[r["publication_group_id"]].append(r)
    return [d[k] for k in sorted(d)]

def fit_m1_v2(real, synthetic=(), start=None):
    groups=group(list(real)+list(synthetic))
    if start is None:
        e0,_=fit_e0(real);start=np.asarray(e0.x)
    starts=(np.array(start),np.array([start[0],-1.,-.5]),np.array([start[0],-3.,.5]))
    bounds=((0.,12.),(-12.,0.),(-4.,3.))
    solutions=[minimize(weighted_objective_grad,x,args=(groups,TAU,FIT_NODES),
               jac=True,method="L-BFGS-B",bounds=bounds,
               options={"maxiter":3000,"ftol":1e-12,"maxls":50}) for x in starts]
    good=[s for s in solutions if s.success and math.isfinite(s.fun) and s.fun<1e299]
    if not good:raise RuntimeError("M1 fit failed: "+",".join(str(s.message) for s in solutions))
    best=min(good,key=lambda s:s.fun)
    check=weighted_objective_grad(best.x,groups,TAU,CHECK_NODES)[0]
    delta=abs(float(best.fun)-check)
    if delta>1e-5:raise RuntimeError(f"Weighted objective quadrature unstable {delta}")
    if not synthetic:
        old=study_marginal_objective(best.x,groups_for([
            dict(r,study_id=r["publication_group_id"]) for r in real]),TAU,FIT_NODES)
        if abs(float(best.fun)-old)>1e-7:raise RuntimeError("M1 original likelihood mismatch")
    return best,delta

def posterior_offset(params, source):
    q,logw=quadrature(FIT_NODES);offset=TAU*q
    alpha,beta,logk=map(float,params);k=math.exp(logk)
    stress=np.array([float(r["stress_amplitude_MPa"]) for r in source])
    z=np.log10([float(r["cycles_failure_or_bound"]) for r in source])
    event=np.array([int(r["failure_event_1"]) for r in source])
    h=LN10*k*(z[:,None]-alpha-beta*np.log2(stress[:,None]/100)-offset[None,:])
    terms=event[:,None]*(logk+math.log(LN10)+h)-np.exp(h)
    p=np.exp(logw+terms.sum(axis=0)-logsumexp(logw+terms.sum(axis=0)))
    return offset,p

def generate(hold,seed,train,params):
    rows=[];ledger=[]
    for source in sorted({r["publication_group_id"] for r in train}):
        original=[r for r in train if r["publication_group_id"]==source]
        if source not in PRIMARY:raise AssertionError(source)
        # All three primary papers document 10^7 as the experiment's stop.
        rng=np.random.default_rng(seed+{"Wu_2021":0,"Chen_2024":1000,"Matusu_2026":2000}[source])
        offsets,p=posterior_offset(params,original)
        b=float(rng.choice(offsets,p=p))
        n=len(original)
        strata=defaultdict(list)
        for r in original:strata[series_key(r)].append(r)
        if source=="Matusu_2026" and len(strata)!=11:
            raise AssertionError("Matušů series mismatch")
        for series,existing in sorted(strata.items()):
            for i in range(len(existing)):
                parent=existing[int(rng.integers(len(existing)))]
                stress=float(parent["stress_amplitude_MPa"])
                rejected=0
                while True:
                    z=float(params[0]+params[1]*math.log2(stress/100)+b+
                            math.log10(rng.exponential())/math.exp(params[2]))
                    if not math.isfinite(z):
                        raise RuntimeError(f"Generated nonfinite life: {hold} {seed} {source}")
                    if z>=0:break
                    rejected+=1
                    if rejected>10000:
                        raise RuntimeError(f"Cannot draw physical life: {hold} {seed} {source}")
                event=int(z<=math.log10(STOP))
                observed=10**z if event else STOP
                rid=f"SYN:{hold}:{seed}:{source}:{series}:{i}"
                generated=dict(record_id=rid,publication_group_id=source,
                    stress_amplitude_MPa=stress,cycles_failure_or_bound=observed,
                    failure_event_1=event,weight=CAP/n)
                rows.append(generated)
                ledger.append(dict(record_id=rid,fold=hold,seed=seed,synthetic=True,
                    training_only=True,publication_group_id=source,
                    platform_family_id=parent["platform_family_id"],series=series,
                    parent_stress_record_id=parent["record_id"],stress_amplitude_MPa=stress,
                    stress_ratio_R=parent["stress_ratio_R"],latent_log10_cycles=z,
                    observed_cycles_or_bound=observed,bound_operator="=" if event else ">=",
                    failure_event_1=event,stop_cycles=STOP,sampled_source_offset_log10=b,
                    weight_S0=CAP/n,weight_S1=CAP/n if source!="Chen_2024" else 0.,
                    rejected_subcycle_draws=rejected))
    if len(rows)!=len(train) or len({r["record_id"] for r in rows})!=len(rows):
        raise AssertionError("Synthetic count or uniqueness")
    return rows,ledger

def predict(params,arm,test,hold,seed,train):
    stress=np.array([float(r["stress_amplitude_MPa"]) for r in test])
    life=np.array([float(r["cycles_failure_or_bound"]) for r in test])
    event=np.array([int(r["failure_event_1"]) for r in test])
    if arm=="C0":
        logterms=e0_logterms(params,stress,life,event);quad=0.
    else:
        logterms,quad=prediction_checks(params,TAU,stress,life,event)
    lo=min(float(r["stress_amplitude_MPa"]) for r in train)
    hi=max(float(r["stress_amplitude_MPa"]) for r in train)
    out=[]
    for i,r in enumerate(test):
        med=(median_e0(params,stress[i]) if arm=="C0" else median_m1(params,TAU,stress[i]))
        nll=-float(logterms[i])
        if not math.isfinite(nll) or not math.isfinite(med):raise RuntimeError(f"Nonfinite prediction {arm} {hold}")
        out.append(dict(arm=arm,fold=hold,seed="" if seed is None else seed,
            record_id=r["record_id"],event=int(event[i]),
            stress_amplitude_MPa=float(stress[i]),cycles_or_bound=float(life[i]),
            stress_outside_train_range=int(stress[i]<lo or stress[i]>hi),
            nll_log10_life_density=nll,median_log10_cycles=float(med),
            failure_abs_error_log10=abs(float(med)-math.log10(life[i])) if event[i] else "",
            quadrature_max_delta=quad))
    return out

def summary(preds):
    grouped=defaultdict(list)
    for r in preds:grouped[(r["arm"],r["seed"],r["fold"])].append(r)
    result=[]
    for (arm,seed,fold),rows in sorted(grouped.items()):
        def mean(a):return sum(a)/len(a) if a else ""
        fail=[r["nll_log10_life_density"] for r in rows if r["event"]]
        run=[r["nll_log10_life_density"] for r in rows if not r["event"]]
        inside=[r["nll_log10_life_density"] for r in rows if not r["stress_outside_train_range"]]
        outside=[r["nll_log10_life_density"] for r in rows if r["stress_outside_train_range"]]
        result.append(dict(arm=arm,seed=seed,fold=fold,n=len(rows),failures=len(fail),
            runouts=len(run),mean_nll=mean([r["nll_log10_life_density"] for r in rows]),
            failure_mean_nll=mean(fail),runout_mean_nll=mean(run),
            within_range_n=len(inside),within_range_mean_nll=mean(inside),
            outside_range_n=len(outside),outside_range_mean_nll=mean(outside),
            failure_median_mae_log10=mean([r["failure_abs_error_log10"] for r in rows if r["event"]])))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--out",type=Path,default=Path("results/development_v2"));a=p.parse_args()
    folds=input_checks(a.root);a.out.mkdir(parents=True,exist_ok=True)
    preds=[];fits=[];ledger=[];problems=[]
    for hold,(train,test) in folds.items():
        e0,_=fit_e0(train)
        teacher,quad=fit_m1_v2(train,start=e0.x)
        for arm,params,sol,delta in (("C0",e0.x,e0,0.),("C1",teacher.x,teacher,quad)):
            preds.extend(predict(params,arm,test,hold,None,train))
            fits.append(dict(arm=arm,fold=hold,seed="",n_real=len(train),n_synthetic=0,
                synthetic_weight=0,alpha=float(params[0]),beta=float(params[1]),
                log_shape=float(params[2]),train_objective=float(sol.fun),
                objective_quad_delta=delta,optimizer_success=bool(sol.success),
                beta_at_constraint=int(params[1]<=-11.999999 or params[1]>=-1e-6),
                shape_at_constraint=int(params[2]<=-3.999999 or params[2]>=2.999999)))
        for seed in SEEDS:
            generated,records=generate(hold,seed,train,teacher.x);ledger.extend(records)
            for arm in ("S0","S1"):
                synthetic=generated if arm=="S0" else [r for r in generated if r["publication_group_id"]!="Chen_2024"]
                try:
                    sol,delta=fit_m1_v2(train,synthetic,start=teacher.x)
                    this=predict(sol.x,arm,test,hold,seed,train)
                except Exception as ex:
                    problems.append(dict(arm=arm,fold=hold,seed=seed,error=repr(ex)))
                    continue
                preds.extend(this)
                fits.append(dict(arm=arm,fold=hold,seed=seed,n_real=len(train),
                    n_synthetic=len(synthetic),synthetic_weight=sum(r["weight"] for r in synthetic),
                    alpha=float(sol.x[0]),beta=float(sol.x[1]),log_shape=float(sol.x[2]),
                    train_objective=float(sol.fun),objective_quad_delta=delta,
                    optimizer_success=bool(sol.success),
                    beta_at_constraint=int(sol.x[1]<=-11.999999 or sol.x[1]>=-1e-6),
                    shape_at_constraint=int(sol.x[2]<=-3.999999 or sol.x[2]>=2.999999)))
            print(hold,seed,"complete",flush=True)
    metrics=summary(preds)
    # Preserve every deterministic and stochastic score; never substitute another seed.
    write(a.out/"predictions_real_test.csv",preds)
    write(a.out/"fold_metrics.csv",metrics)
    write(a.out/"fit_parameters.csv",fits)
    write(a.out/"synthetic_training_ledger.csv",ledger)
    a.out.joinpath("fit_problems.json").write_text(json.dumps(problems,indent=2)+"\n")
    # Secondary challenge: an unseen R=-1 publication, never used to choose
    # generation rules or the primary stress-only model.
    cohort=read(a.root/COHORT)
    train_all=[r for r in cohort if r["publication_group_id"] in PRIMARY]
    romano=[r for r in cohort if r["publication_group_id"]=="Romano_2018"]
    e0_all,_=fit_e0(train_all)
    teacher_all,quad_all=fit_m1_v2(train_all,start=e0_all.x)
    transfer_pred=[];transfer_fit=[];transfer_ledger=[]
    for arm,sol,delta in (("C0",e0_all,0.),("C1",teacher_all,quad_all)):
        transfer_pred.extend(predict(sol.x,arm,romano,"Romano_2018_transfer",None,train_all))
        transfer_fit.append(dict(arm=arm,fold="Romano_2018_transfer",seed="",
            n_real=len(train_all),n_synthetic=0,synthetic_weight=0,
            alpha=float(sol.x[0]),beta=float(sol.x[1]),log_shape=float(sol.x[2]),
            train_objective=float(sol.fun),objective_quad_delta=delta,
            optimizer_success=bool(sol.success),
            beta_at_constraint=int(sol.x[1]<=-11.999999 or sol.x[1]>=-1e-6),
            shape_at_constraint=int(sol.x[2]<=-3.999999 or sol.x[2]>=2.999999)))
    for seed in SEEDS:
        generated,records=generate("Romano_2018_transfer",seed,train_all,teacher_all.x)
        transfer_ledger.extend(records)
        for arm in ("S0","S1"):
            synthetic=generated if arm=="S0" else [r for r in generated if r["publication_group_id"]!="Chen_2024"]
            try:
                sol,delta=fit_m1_v2(train_all,synthetic,start=teacher_all.x)
                transfer_pred.extend(predict(sol.x,arm,romano,"Romano_2018_transfer",seed,train_all))
            except Exception as ex:
                problems.append(dict(arm=arm,fold="Romano_2018_transfer",seed=seed,error=repr(ex)))
                continue
            transfer_fit.append(dict(arm=arm,fold="Romano_2018_transfer",seed=seed,
                n_real=len(train_all),n_synthetic=len(synthetic),
                synthetic_weight=sum(r["weight"] for r in synthetic),
                alpha=float(sol.x[0]),beta=float(sol.x[1]),log_shape=float(sol.x[2]),
                train_objective=float(sol.fun),objective_quad_delta=delta,
                optimizer_success=bool(sol.success),
                beta_at_constraint=int(sol.x[1]<=-11.999999 or sol.x[1]>=-1e-6),
                shape_at_constraint=int(sol.x[2]<=-3.999999 or sol.x[2]>=2.999999)))
    write(a.out/"romano_transfer_predictions.csv",transfer_pred)
    write(a.out/"romano_transfer_metrics.csv",summary(transfer_pred))
    write(a.out/"romano_transfer_fit_parameters.csv",transfer_fit)
    write(a.out/"romano_transfer_synthetic_training_ledger.csv",transfer_ledger)
    a.out.joinpath("fit_problems.json").write_text(json.dumps(problems,indent=2)+"\n")
    metadata=dict(protocol="docs/development_v2_precomparison_protocol_2026-10-06.md",
        input_blobs=BLOBS,seeds=SEEDS,tau=TAU,stop=STOP,
        effective_synthetic_weight_per_source=CAP,fit_nodes=FIT_NODES,
        check_nodes=CHECK_NODES,software=dict(python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__),fit_problems=problems,
        cohort_is_original_experimental_only=True,synthetic_is_training_only=True)
    (a.out/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"fold_metrics":metrics,"romano_transfer_metrics":summary(transfer_pred),
                      "fit_problems":problems},allow_nan=False))

if __name__=="__main__":main()
