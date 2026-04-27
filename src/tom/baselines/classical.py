#!/usr/bin/env python3
"""
17_classical_baselines.py — Classical candidate-set baselines for Exp04.

Inputs are inference records from 12_infer_vllm.py with saved `candidates`.
Outputs mimic the LoRA prediction JSONL schema so 14_temporal_eval.py can score
and plot them without a separate code path.
"""

import argparse
import json
import math
import os


METHODS = {"frequency", "gravity", "huff", "mnl_grid", "mnl_rich",
           "deep_gravity", "lgb_rank", "xgb_rank"}

POLYGON_TYPES = ["OWNED", "SHARED_BUILDING", "SHARED_DISTINCT"]  # FALLBACK is reference


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def load_records(path, max_records=0):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("candidates"):
                records.append(rec)
            if max_records > 0 and len(records) >= max_records:
                break
    return records


def utilities(candidates, method, alpha=1.0, beta=1.0, distance_offset_m=100.0):
    vals = []
    for cand in candidates:
        freq = max(float(cand.get("cluster_freq", 0) or 0), 0.0)
        dist = max(float(cand.get("dist_m", 0.0) or 0.0), 0.0) + distance_offset_m

        if method == "frequency":
            util = math.log1p(freq)
        elif method == "gravity":
            util = math.log1p(freq) - beta * math.log(dist)
        elif method == "huff":
            # Huff (1963) / MCI: multiplicative attraction, power-law distance.
            util = alpha * math.log1p(freq) - beta * math.log(dist)
        elif method == "mnl_grid":
            # McFadden (1974) / Ben-Akiva & Lerman (1985): linear distance in
            # utility → exponential distance decay. beta is cost per km.
            util = alpha * math.log1p(freq) - beta * (dist / 1000.0)
        else:
            raise ValueError(f"Unknown method: {method}")
        vals.append(util)
    return vals


def softmax(vals):
    if not vals:
        return []
    max_val = max(vals)
    exp_vals = [math.exp(v - max_val) for v in vals]
    total = sum(exp_vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / total for v in exp_vals]


def nll(records, method, alpha, beta, distance_offset_m):
    total = 0.0
    n = 0
    for rec in records:
        candidates = rec.get("candidates") or []
        true_id = rec.get("true_poi_id")
        true_idx = None
        for idx, cand in enumerate(candidates):
            if cand.get("poi_id") == true_id:
                true_idx = idx
                break
        if true_idx is None:
            continue
        probs = softmax(utilities(candidates, method, alpha, beta, distance_offset_m))
        total += -math.log(max(probs[true_idx], 1e-12))
        n += 1
    return total / max(n, 1), n


def fit_mnl_continuous(records, distance_offset_m, method="mnl_grid", max_iter=200):
    """Continuous (α, β) fit via L-BFGS-B minimising NLL under the method's own utility.
    Replaces the old grid search (BUG-028: grid boundary hits). Pass method="huff"
    for power-law log-distance; "mnl_grid" for linear km."""
    from scipy.optimize import minimize

    def obj(params):
        a, b = float(params[0]), float(params[1])
        loss, _ = nll(records, method, a, b, distance_offset_m)
        return loss

    result = minimize(
        obj,
        x0=[1.0, 1.0],
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={"maxiter": max_iter},
    )
    _, n_fit = nll(records, method, float(result.x[0]), float(result.x[1]), distance_offset_m)
    return {
        "alpha": float(result.x[0]),
        "beta": float(result.x[1]),
        "nll": float(result.fun),
        "n_fit": n_fit,
        "converged": bool(result.success),
    }


def fit_gravity_continuous(records, distance_offset_m, max_iter=200):
    """Continuous β fit (α fixed at 1) for gravity — null-reference to huff."""
    from scipy.optimize import minimize_scalar

    def obj(beta):
        loss, _ = nll(records, "gravity", 1.0, float(beta), distance_offset_m)
        return loss

    result = minimize_scalar(
        obj,
        bounds=(0.0, 20.0),
        method="bounded",
        options={"maxiter": max_iter, "xatol": 1e-4},
    )
    _, n_fit = nll(records, "gravity", 1.0, float(result.x), distance_offset_m)
    return {
        "alpha": 1.0,
        "beta": float(result.x),
        "nll": float(result.fun),
        "n_fit": n_fit,
        "converged": bool(result.success),
    }


# ── mnl_rich: conditional logit with polygon_type / top-K sub_category /
#            competing-destinations feature (Fotheringham 1983). Fit via
#            scipy L-BFGS-B on the analytic negative log-likelihood gradient. ──

def _top_k_subcategories(records, k):
    from collections import Counter
    counter = Counter()
    for rec in records:
        for c in rec.get("candidates") or []:
            counter[str(c.get("sub_category") or "").strip()] += 1
    return [cat for cat, _ in counter.most_common(k) if cat]


def _rich_feature_matrix(candidates, top_subcats, distance_offset_m):
    """Return (N_cand, D) feature matrix for a single choice set."""
    import numpy as np
    n = len(candidates)
    cat_idx = {cat: i for i, cat in enumerate(top_subcats)}
    D = 2 + len(POLYGON_TYPES) + len(top_subcats) + 1

    dists = [max(float(c.get("dist_m", 0.0) or 0.0), 0.0) for c in candidates]
    # competing destinations: count candidates with smaller dist in same set
    order = sorted(range(n), key=lambda i: dists[i])
    n_closer = [0] * n
    for rank, i in enumerate(order):
        n_closer[i] = rank

    X = np.zeros((n, D), dtype=np.float64)
    for i, c in enumerate(candidates):
        freq = max(float(c.get("cluster_freq", 0) or 0), 0.0)
        dist = dists[i] + distance_offset_m
        X[i, 0] = math.log1p(freq)
        X[i, 1] = math.log(dist)
        poly = str(c.get("polygon_type") or "")
        for k_pt, pt in enumerate(POLYGON_TYPES):
            if poly == pt:
                X[i, 2 + k_pt] = 1.0
        cat = str(c.get("sub_category") or "").strip()
        if cat in cat_idx:
            X[i, 2 + len(POLYGON_TYPES) + cat_idx[cat]] = 1.0
        X[i, -1] = math.log1p(n_closer[i])
    return X


def _build_fit_tensors(records, top_subcats, distance_offset_m):
    """Concatenate feature matrices across fit records; keep per-record boundaries."""
    import numpy as np
    Xs = []
    true_positions = []
    sizes = []
    for rec in records:
        candidates = rec.get("candidates") or []
        true_pid = rec.get("true_poi_id")
        true_idx = next((i for i, c in enumerate(candidates) if c.get("poi_id") == true_pid), None)
        if true_idx is None or not candidates:
            continue
        X = _rich_feature_matrix(candidates, top_subcats, distance_offset_m)
        Xs.append(X)
        true_positions.append(true_idx)
        sizes.append(len(candidates))
    if not Xs:
        return None
    X_all = np.concatenate(Xs, axis=0)
    offsets = np.cumsum([0] + sizes[:-1])
    return {
        "X": X_all,
        "offsets": np.array(offsets, dtype=np.int64),
        "sizes": np.array(sizes, dtype=np.int64),
        "true_pos": np.array(true_positions, dtype=np.int64),
    }


def fit_mnl_rich(records, top_subcats, distance_offset_m, max_iter=200, l2_lambda=1e-3):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    data = _build_fit_tensors(records, top_subcats, distance_offset_m)
    if data is None:
        return None
    # NOTE on β sign: in this distance-aware candidate set, truth POIs are
    # systematically farther than filler candidates (sampler pads with nearby
    # FALLBACK POIs). Both `log(dist+off)` and `log1p(n_closer)` receive
    # positive β under MLE; their combined effect gives valid predictions, but
    # individual coefficients should NOT be read as "distance deters".
    X = data["X"]
    offsets = data["offsets"]
    sizes = data["sizes"]
    true_pos = data["true_pos"]
    D = X.shape[1]
    # Ridge penalty on dummy coefficients only (skip first 2 = log1p(freq), log(dist))
    # to prevent rare categorical quasi-separation; does not regularise continuous features.
    ridge_mask = np.ones(D, dtype=np.float64); ridge_mask[:2] = 0.0

    def neg_ll_and_grad(beta):
        logits = X @ beta
        nll = 0.0
        grad = np.zeros(D, dtype=np.float64)
        for i in range(len(sizes)):
            s = offsets[i]
            e = s + sizes[i]
            block = logits[s:e]
            lse = logsumexp(block)
            true_global = s + true_pos[i]
            nll += -(logits[true_global] - lse)
            p = np.exp(block - lse)
            grad -= X[true_global]
            grad += p @ X[s:e]
        # Ridge on dummies only
        nll += 0.5 * l2_lambda * float(((beta ** 2) * ridge_mask).sum())
        grad += l2_lambda * beta * ridge_mask
        return nll, grad

    beta0 = np.zeros(D, dtype=np.float64)
    result = minimize(
        neg_ll_and_grad,
        beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "disp": False},
    )
    # Report mean NLL without the ridge penalty for comparability across fits.
    penalty = 0.5 * l2_lambda * float(((np.asarray(result.x) ** 2) * ridge_mask).sum())
    ll_only = float(result.fun) - penalty
    return {
        "beta": result.x.tolist(),
        "feature_names": (
            ["log1p(cluster_freq)", "log(dist+off)"]
            + [f"polygon={pt}" for pt in POLYGON_TYPES]
            + [f"subcat={cat}" for cat in top_subcats]
            + ["log1p(n_closer)"]
        ),
        "top_subcats": list(top_subcats),
        "distance_offset_m": float(distance_offset_m),
        "l2_lambda": float(l2_lambda),
        "mean_nll": ll_only / max(len(sizes), 1),
        "n_fit": int(len(sizes)),
        "converged": bool(result.success),
        "message": str(result.message),
    }


def predict_record_mnl_rich(rec, fit, distance_offset_m):
    import numpy as np
    candidates = rec.get("candidates") or []
    if not candidates:
        return None
    # Prefer the fit's stored offset so serialized fits stay self-consistent.
    offset = float(fit.get("distance_offset_m", distance_offset_m))
    X = _rich_feature_matrix(candidates, fit["top_subcats"], offset)
    beta = np.asarray(fit["beta"], dtype=np.float64)
    logits = X @ beta
    max_val = float(logits.max())
    exp_vals = np.exp(logits - max_val)
    total = exp_vals.sum()
    probs = (exp_vals / total).tolist() if total > 0 else [1.0 / len(candidates)] * len(candidates)

    ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
    top = ranked[:5]
    top_total = sum(p for _, p in top) or 1.0
    top_5 = [
        {"poi": cand["poi_id"], "prob": round(p / top_total, 4)}
        for cand, p in top
    ]
    return {
        "index": rec.get("index"),
        "cuebiq_id": rec.get("cuebiq_id"),
        "cluster": rec.get("cluster"),
        "admin2_id": rec.get("admin2_id"),
        "stop_ts": rec.get("stop_ts"),
        "work_cbg": rec.get("work_cbg", ""),
        "true_poi_id": rec.get("true_poi_id"),
        "true_polygon_type": rec.get("true_polygon_type", ""),
        "origin": {
            "lat": rec.get("origin_lat"),
            "lon": rec.get("origin_lon"),
            "mode": rec.get("origin_mode"),
            "source": rec.get("origin_source"),
        },
        "next_choice": top_5[0]["poi"] if top_5 else None,
        "top_5": top_5,
        "parse_ok": bool(top_5),
        "baseline_method": "mnl_rich",
        "candidates": candidates,
    }


# ── deep_gravity: MLP utility trained with conditional logit loss (Pappalardo 2021
#   adapted from flow prediction to candidate choice). ──

def _fit_data_matrices(records, top_subcats, distance_offset_m):
    """Return list of (X_rec, true_idx) where X_rec is (n_cand, D)."""
    import numpy as np
    items = []
    for rec in records:
        candidates = rec.get("candidates") or []
        true_pid = rec.get("true_poi_id")
        true_idx = next((i for i, c in enumerate(candidates) if c.get("poi_id") == true_pid), None)
        if true_idx is None or not candidates:
            continue
        X = _rich_feature_matrix(candidates, top_subcats, distance_offset_m)
        items.append((X.astype(np.float32), true_idx))
    return items


def fit_deep_gravity(records, top_subcats, distance_offset_m, hidden=64, epochs=30,
                    batch_size=512, lr=1e-3, weight_decay=1e-5):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    items = _fit_data_matrices(records, top_subcats, distance_offset_m)
    if not items:
        return None

    # All choice sets assumed same size (k=30). Stack into a tensor.
    k = items[0][0].shape[0]
    D = items[0][0].shape[1]
    assert all(x.shape == (k, D) for x, _ in items), "deep_gravity expects uniform candidate-set size"
    X_all = torch.from_numpy(np.stack([x for x, _ in items], axis=0))  # (N, k, D)
    y_all = torch.tensor([i for _, i in items], dtype=torch.long)       # (N,)

    torch.manual_seed(0)
    mlp = nn.Sequential(
        nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    N = X_all.shape[0]
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            xb = X_all[idx]          # (b, k, D)
            yb = y_all[idx]          # (b,)
            logits = mlp(xb).squeeze(-1)   # (b, k)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        mean_loss = total_loss / N
        history.append(mean_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:2d}/{epochs}  mean_nll={mean_loss:.4f}")

    return {
        "model_state": {k: v.detach().clone() for k, v in mlp.state_dict().items()},
        "arch": {"d_in": D, "hidden": hidden},
        "top_subcats": list(top_subcats),
        "distance_offset_m": float(distance_offset_m),
        "mean_nll": history[-1],
        "n_fit": N,
        "epochs": epochs,
    }


def _deep_gravity_mlp(arch):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(arch["d_in"], arch["hidden"]), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(arch["hidden"], arch["hidden"]), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(arch["hidden"], 1),
    )


def predict_record_deep_gravity(rec, fit, distance_offset_m):
    import numpy as np
    import torch
    candidates = rec.get("candidates") or []
    if not candidates:
        return None
    X = _rich_feature_matrix(candidates, fit["top_subcats"], distance_offset_m)
    mlp = _deep_gravity_mlp(fit["arch"])
    mlp.load_state_dict(fit["model_state"])
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.from_numpy(X.astype(np.float32))).squeeze(-1).numpy()
    max_val = float(logits.max())
    exp_vals = np.exp(logits - max_val)
    probs = (exp_vals / exp_vals.sum()).tolist()

    ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
    top = ranked[:5]
    top_total = sum(p for _, p in top) or 1.0
    top_5 = [{"poi": c["poi_id"], "prob": round(p / top_total, 4)} for c, p in top]
    return _build_row(rec, top_5, "deep_gravity", candidates)


# ── lgb_rank / xgb_rank: Learning-to-rank with LambdaMART-style gradient boosting. ──

def _stack_records_for_ranker(records, top_subcats, distance_offset_m):
    import numpy as np
    X_blocks = []
    y_blocks = []
    groups = []
    for rec in records:
        candidates = rec.get("candidates") or []
        true_pid = rec.get("true_poi_id")
        true_idx = next((i for i, c in enumerate(candidates) if c.get("poi_id") == true_pid), None)
        if true_idx is None or not candidates:
            continue
        X = _rich_feature_matrix(candidates, top_subcats, distance_offset_m).astype(np.float32)
        y = np.zeros(len(candidates), dtype=np.int32)
        y[true_idx] = 1
        X_blocks.append(X)
        y_blocks.append(y)
        groups.append(len(candidates))
    if not X_blocks:
        return None
    return {
        "X": np.concatenate(X_blocks, axis=0),
        "y": np.concatenate(y_blocks, axis=0),
        "group": np.array(groups, dtype=np.int32),
    }


def fit_lgb_rank(records, top_subcats, distance_offset_m, n_estimators=500,
                 learning_rate=0.05, num_leaves=63, max_depth=-1):
    import lightgbm as lgb
    data = _stack_records_for_ranker(records, top_subcats, distance_offset_m)
    if data is None:
        return None
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        random_state=0,
        verbose=-1,
    )
    model.fit(data["X"], data["y"], group=data["group"])
    print(f"    best_iter=?  n_features={data['X'].shape[1]}  n_groups={len(data['group'])}  n_rows={data['X'].shape[0]}")
    return {
        "model": model,
        "top_subcats": list(top_subcats),
        "distance_offset_m": float(distance_offset_m),
        "n_fit": int(len(data["group"])),
    }


def fit_xgb_rank(records, top_subcats, distance_offset_m, n_estimators=500,
                 learning_rate=0.05, max_depth=6):
    import xgboost as xgb
    data = _stack_records_for_ranker(records, top_subcats, distance_offset_m)
    if data is None:
        return None
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        tree_method="hist",
        random_state=0,
        verbosity=0,
    )
    model.fit(data["X"], data["y"], group=data["group"])
    print(f"    n_features={data['X'].shape[1]}  n_groups={len(data['group'])}  n_rows={data['X'].shape[0]}")
    return {
        "model": model,
        "top_subcats": list(top_subcats),
        "distance_offset_m": float(distance_offset_m),
        "n_fit": int(len(data["group"])),
    }


def predict_record_ranker(rec, fit, method_name, distance_offset_m):
    import numpy as np
    candidates = rec.get("candidates") or []
    if not candidates:
        return None
    X = _rich_feature_matrix(candidates, fit["top_subcats"], distance_offset_m).astype(np.float32)
    scores = fit["model"].predict(X)
    # Convert scores to "probabilities" via softmax for schema compatibility.
    max_val = float(np.max(scores))
    exp_vals = np.exp(scores - max_val)
    probs = (exp_vals / exp_vals.sum()).tolist()

    ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
    top = ranked[:5]
    top_total = sum(p for _, p in top) or 1.0
    top_5 = [{"poi": c["poi_id"], "prob": round(p / top_total, 4)} for c, p in top]
    row = _build_row(rec, top_5, method_name, candidates)
    # Hybrid-decode hook: expose raw per-candidate scores so downstream scripts
    # can use the ranker over all 30 candidates, not just the top-5.
    row["candidate_scores"] = [
        {"poi": c["poi_id"], "score": float(s), "prob": float(p)}
        for c, s, p in zip(candidates, scores.tolist(), probs)
    ]
    return row


def _build_row(rec, top_5, method_name, candidates):
    return {
        "index": rec.get("index"),
        "cuebiq_id": rec.get("cuebiq_id"),
        "cluster": rec.get("cluster"),
        "admin2_id": rec.get("admin2_id"),
        "stop_ts": rec.get("stop_ts"),
        "work_cbg": rec.get("work_cbg", ""),
        "true_poi_id": rec.get("true_poi_id"),
        "true_polygon_type": rec.get("true_polygon_type", ""),
        "origin": {
            "lat": rec.get("origin_lat"),
            "lon": rec.get("origin_lon"),
            "mode": rec.get("origin_mode"),
            "source": rec.get("origin_source"),
        },
        "next_choice": top_5[0]["poi"] if top_5 else None,
        "top_5": top_5,
        "parse_ok": bool(top_5),
        "baseline_method": method_name,
        "candidates": candidates,
    }


def predict_record(rec, method, alpha, beta, distance_offset_m):
    candidates = rec.get("candidates") or []
    vals = utilities(candidates, method, alpha, beta, distance_offset_m)
    probs = softmax(vals)
    ranked = sorted(
        zip(candidates, probs, vals),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    top = ranked[:5]
    top_total = sum(prob for _, prob, _ in top)
    if top_total <= 0:
        top_probs = [1.0 / len(top)] * len(top) if top else []
    else:
        top_probs = [prob / top_total for _, prob, _ in top]

    top_5 = [
        {"poi": cand["poi_id"], "prob": round(prob, 4)}
        for (cand, _, _), prob in zip(top, top_probs)
    ]
    return {
        "index": rec.get("index"),
        "cuebiq_id": rec.get("cuebiq_id"),
        "cluster": rec.get("cluster"),
        "admin2_id": rec.get("admin2_id"),
        "stop_ts": rec.get("stop_ts"),
        "work_cbg": rec.get("work_cbg", ""),
        "true_poi_id": rec.get("true_poi_id"),
        "true_polygon_type": rec.get("true_polygon_type", ""),
        "origin": {
            "lat": rec.get("origin_lat"),
            "lon": rec.get("origin_lon"),
            "mode": rec.get("origin_mode"),
            "source": rec.get("origin_source"),
        },
        "next_choice": top_5[0]["poi"] if top_5 else None,
        "top_5": top_5,
        "parse_ok": bool(top_5),
        "baseline_method": method,
        "baseline_alpha": alpha,
        "baseline_beta": beta,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_in", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), default="frequency")
    parser.add_argument("--fit_records_in", default="")
    parser.add_argument(
        "--fit_on_records_in",
        action="store_true",
        help="Fit on --records_in when no separate fit set is available. "
             "Use only for in-sample diagnostics, not temporal-OOS scoring.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--distance_offset_m", type=float, default=100.0)
    parser.add_argument("--max_fit_records", type=int, default=50000)
    parser.add_argument("--mnl_rich_top_subcats", type=int, default=15,
                        help="Top-K sub_category one-hot dims for mnl_rich")
    parser.add_argument("--mnl_rich_max_iter", type=int, default=200)
    parser.add_argument("--l2_lambda", type=float, default=1e-3,
                        help="Ridge penalty on mnl_rich dummy coefficients "
                             "(skips log1p(freq)/log(dist)). 0 = off.")
    args = parser.parse_args()

    print("=== Classical Baseline ===")
    print(f"  method: {args.method}")
    print(f"  records_in: {args.records_in}")

    alpha = args.alpha
    beta = args.beta
    fit_summary = None
    rich_fit = None

    if args.method == "mnl_rich":
        if not args.fit_records_in and not args.fit_on_records_in:
            raise RuntimeError("mnl_rich requires --fit_records_in or --fit_on_records_in")
        fit_path = args.fit_records_in or args.records_in
        fit_records = load_records(fit_path, max_records=args.max_fit_records)
        print(f"  fit records: {len(fit_records):,}")
        top_subcats = _top_k_subcategories(fit_records, args.mnl_rich_top_subcats)
        print(f"  top-{len(top_subcats)} sub_categories: {top_subcats[:5]}... (+{max(0,len(top_subcats)-5)} more)")
        rich_fit = fit_mnl_rich(fit_records, top_subcats, args.distance_offset_m,
                                args.mnl_rich_max_iter, l2_lambda=args.l2_lambda)
        print(f"  converged={rich_fit['converged']} mean_nll={rich_fit['mean_nll']:.4f} n={rich_fit['n_fit']:,}")
        print(f"  top |β| features:")
        beta_arr = list(enumerate(rich_fit["beta"]))
        beta_arr.sort(key=lambda x: -abs(x[1]))
        for i, b in beta_arr[:8]:
            print(f"    {rich_fit['feature_names'][i]:<28s} β={b:+.4f}")
        fit_summary = {
            "method": "mnl_rich",
            "n_fit": rich_fit["n_fit"],
            "nll": rich_fit["mean_nll"],
            "converged": rich_fit["converged"],
            "feature_names": rich_fit["feature_names"],
            "beta": rich_fit["beta"],
        }
    elif args.method in {"deep_gravity", "lgb_rank", "xgb_rank"}:
        if not args.fit_records_in and not args.fit_on_records_in:
            raise RuntimeError(f"{args.method} requires --fit_records_in or --fit_on_records_in")
        fit_path = args.fit_records_in or args.records_in
        fit_records = load_records(fit_path, max_records=args.max_fit_records)
        top_subcats = _top_k_subcategories(fit_records, args.mnl_rich_top_subcats)
        print(f"  fit records: {len(fit_records):,}")
        print(f"  top-{len(top_subcats)} sub_categories")
        if args.method == "deep_gravity":
            rich_fit = fit_deep_gravity(fit_records, top_subcats, args.distance_offset_m)
            fit_summary = {"method": "deep_gravity", "n_fit": rich_fit["n_fit"],
                           "mean_nll": rich_fit["mean_nll"], "arch": rich_fit["arch"],
                           "epochs": rich_fit["epochs"]}
        elif args.method == "lgb_rank":
            rich_fit = fit_lgb_rank(fit_records, top_subcats, args.distance_offset_m)
            fit_summary = {"method": "lgb_rank", "n_fit": rich_fit["n_fit"]}
        else:
            rich_fit = fit_xgb_rank(fit_records, top_subcats, args.distance_offset_m)
            fit_summary = {"method": "xgb_rank", "n_fit": rich_fit["n_fit"]}
    else:
        should_fit = args.method in {"gravity", "huff", "mnl_grid"} and (
            bool(args.fit_records_in) or args.fit_on_records_in
        )
        if should_fit:
            fit_path = args.fit_records_in or args.records_in
            fit_records = load_records(fit_path, max_records=args.max_fit_records)
            if args.method == "gravity":
                fit_summary = fit_gravity_continuous(fit_records, args.distance_offset_m)
            else:
                # method-aware: huff uses its own power-law NLL, mnl_grid uses
                # linear-distance NLL (BUG-033 prevention).
                fit_summary = fit_mnl_continuous(
                    fit_records, args.distance_offset_m, method=args.method,
                )
            alpha = fit_summary["alpha"]
            beta = fit_summary["beta"]
            print(f"  fitted alpha={alpha:.4f}, beta={beta:.4f}, "
                  f"nll={fit_summary['nll']:.4f}, n={fit_summary['n_fit']:,}, "
                  f"converged={fit_summary.get('converged')}")

    records = load_records(args.records_in)
    if not records:
        raise RuntimeError(
            f"No usable records with candidates found in {args.records_in}. "
            "Rebuild inference records with the updated 12_infer_vllm.py."
        )
    print(f"  predict records: {len(records):,}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            if args.method == "mnl_rich":
                row = predict_record_mnl_rich(rec, rich_fit, args.distance_offset_m)
            elif args.method == "deep_gravity":
                row = predict_record_deep_gravity(rec, rich_fit, args.distance_offset_m)
            elif args.method in {"lgb_rank", "xgb_rank"}:
                row = predict_record_ranker(rec, rich_fit, args.method, args.distance_offset_m)
            else:
                row = predict_record(rec, args.method, alpha, beta, args.distance_offset_m)
            if row is None:
                continue
            row["index"] = idx
            if fit_summary:
                row["baseline_fit"] = fit_summary
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  saved: {args.out}")


if __name__ == "__main__":
    main()
