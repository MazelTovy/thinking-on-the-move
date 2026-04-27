"""Poster-ready version of `temporal_scatter_4panel.png`.

Differences vs `14_temporal_eval.py:plot_temporal_scatter_4panel`:
 - Figure 20x16 in @ 300 dpi (from 14x12 @ default), print-safe.
 - All text ~2.2x larger (titles, labels, ticks, legend).
 - Marker size scales with **actual visit mass** (user's ask): the POIs
   that matter visually pop out; the long tail stays visible but small.
 - Emphasis markers for New/Disappeared POIs (bigger + thicker edge).
 - Thicker, clearer diagonal reference line + y=x shading.
 - Top-10 POI labels use leader lines instead of overlapping boxes.
 - Output: experiments/<exp>/eval/figures/poster_temporal_scatter.{png,pdf}

Reuses data loaders from 14_temporal_eval.py.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, "/scratch/sx2490/econai/nyc_metro")
import importlib.util
spec = importlib.util.spec_from_file_location("t14", "/scratch/sx2490/econai/nyc_metro/14_temporal_eval.py")
t14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t14)


def size_for_mass(mass, mass_ref, n_trips, base=1, cap=1300):
    """Smooth linear growth: tail POIs (1-5 visits) stay near invisible,
    top POI reaches `cap`. No plateau — everything is progressive."""
    visits = np.asarray(mass, dtype=float) * n_trips
    max_visits = max(mass_ref * n_trips, 1.0)
    s = base + (cap - base) * np.clip(visits / max_visits, 0.0, 1.0)
    return s


def draw_poster_panel(ax, actual, predicted, temporal_status_by_pid, poi_meta,
                      title, n_trips, log_mode=False):
    """Axes are in absolute visit-count units (mass * n_trips).
    Scaling uses mass internally but labels/ticks display as integer visits."""
    all_pois = list(set(actual) | set(predicted))
    mass_ref = max(max(actual.values(), default=0.001),
                   max(predicted.values(), default=0.001))
    # Visit-count axis limits
    log_lo_count = 0.5        # sub-visit lower floor for log scale
    log_hi_count = max(mass_ref * n_trips * 1.2, 100)

    stable, new_list, dis_list = [], [], []
    for pid in all_pois:
        a = actual.get(pid, 0.0); p = predicted.get(pid, 0.0)
        a_cnt = a * n_trips; p_cnt = p * n_trips
        if log_mode:
            if a == 0 and p == 0: continue
            a_cnt = max(a_cnt, log_lo_count); p_cnt = max(p_cnt, log_lo_count)
        group = t14.temporal_group(temporal_status_by_pid.get(pid, "unknown"))
        entry = (pid, a_cnt, p_cnt, max(a, p))
        if group == "new": new_list.append(entry)
        elif group == "disappeared": dis_list.append(entry)
        else: stable.append(entry)

    # Helpers
    def xy_s(lst):
        if not lst: return [], [], []
        xs = [e[1] for e in lst]; ys = [e[2] for e in lst]
        masses = [e[3] for e in lst]
        return xs, ys, size_for_mass(masses, mass_ref, n_trips)

    # All three groups use circles; only the color differs.
    sx, sy, ss = xy_s(stable)
    if sx:
        ax.scatter(sx, sy, s=ss, alpha=0.50, c="#4C72B0",
                   edgecolors="none", label=f"Stable ({len(sx):,})",
                   zorder=3, rasterized=True)
    nx, ny, ns = xy_s(new_list)
    if nx:
        ax.scatter(nx, ny, s=ns, alpha=0.85, c="#C44E52",
                   edgecolors="none",
                   label=f"New ({len(nx):,})", zorder=5)
    dx, dy, ds = xy_s(dis_list)
    if dx:
        ax.scatter(dx, dy, s=ds, alpha=0.85, c="#DD8452",
                   edgecolors="none",
                   label=f"Disappeared ({len(dx):,})", zorder=5)

    # Diagonal + axes in visit-count units
    if log_mode:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(log_lo_count, log_hi_count); ax.set_ylim(log_lo_count, log_hi_count)
        ax.plot([log_lo_count, log_hi_count], [log_lo_count, log_hi_count],
                color="black", ls="--", lw=2.2, alpha=0.45, zorder=2, label="y = x")
        fac = 10 ** 0.04
        ax.fill_between([log_lo_count, log_hi_count],
                        [log_lo_count * fac, log_hi_count * fac],
                        [log_lo_count / fac, log_hi_count / fac],
                        color="gray", alpha=0.10, zorder=1)
        ax.set_xlabel("Actual visits (OOS trips to POI)", fontsize=24, labelpad=12)
        ax.set_ylabel("Predicted top-1 visits (sum of model mass)", fontsize=24, labelpad=12)
    else:
        hi = mass_ref * n_trips * 1.1
        ax.plot([0, hi], [0, hi], color="black", ls="--", lw=2.2,
                alpha=0.45, zorder=2, label="y = x")
        ax.set_xlim(-hi * 0.02, hi); ax.set_ylim(-hi * 0.02, hi)
        ax.set_xlabel("Actual visits (OOS trips to POI)", fontsize=24, labelpad=12)
        ax.set_ylabel("Predicted top-1 visits (sum of model mass)", fontsize=24, labelpad=12)

    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(title, fontsize=26, pad=16)
    ax.legend(loc="upper left", fontsize=17, framealpha=0.92,
              markerscale=1.0, labelspacing=0.6)
    ax.grid(alpha=0.25, lw=0.5, zorder=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="exp02_wcbg")
    p.add_argument("--out_stem", default="poster_temporal_scatter")
    args = p.parse_args()

    BASE = "/scratch/sx2490/econai/nyc_metro"
    exp_dir = Path(BASE) / "experiments" / args.exp
    fig_dir = exp_dir / "eval" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    class _A:
        exp = args.exp
        snapshot_manifest = None
        cbg_poi_dir = None
        fallback_poi_csv = None
    manifest_path, cbg_poi_dir, fallback = t14.resolve_snapshot_paths(str(exp_dir), _A())
    print(f"cbg_poi_dir={cbg_poi_dir}  fallback={fallback}")

    valid_set, name_to_valid, poi_meta = t14.build_valid_poi_set(cbg_poi_dir, fallback)
    temporal_status_by_pid = t14.load_temporal_status_map(valid_set, name_to_valid)

    pred_train = exp_dir / "pred_insample_2021.jsonl"
    pred_test = exp_dir / "pred_oos_2022.jsonl"
    train_eval = t14.load_visit_level_eval(str(pred_train), valid_set, name_to_valid)
    test_eval = t14.load_visit_level_eval(str(pred_test), valid_set, name_to_valid)

    train_actual = train_eval["actual_top1"]
    train_pred = train_eval["predicted_top1"]
    test_actual = test_eval["actual_top1"]
    test_pred = test_eval["predicted_top1"]

    n_train_trips = train_eval["stats"]["records_with_predicted_top5"]
    n_test_trips = test_eval["stats"]["records_with_predicted_top5"]
    print(f"n_train_trips={n_train_trips}  n_test_trips={n_test_trips}")

    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    draw_poster_panel(
        axes[0, 0], train_actual, train_pred, temporal_status_by_pid, poi_meta,
        "In-sample 2021 val — linear", n_train_trips, log_mode=False,
    )
    draw_poster_panel(
        axes[0, 1], test_actual, test_pred, temporal_status_by_pid, poi_meta,
        "OOS 2022 Q1 — linear", n_test_trips, log_mode=False,
    )
    draw_poster_panel(
        axes[1, 0], train_actual, train_pred, temporal_status_by_pid, poi_meta,
        "In-sample 2021 val — log", n_train_trips, log_mode=True,
    )
    draw_poster_panel(
        axes[1, 1], test_actual, test_pred, temporal_status_by_pid, poi_meta,
        "OOS 2022 Q1 — log", n_test_trips, log_mode=True,
    )
    fig.suptitle("Actual vs Predicted Visits  (point size ∝ actual visit count)",
                 fontsize=30, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = fig_dir / f"{args.out_stem}.png"
    out_pdf = fig_dir / f"{args.out_stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
