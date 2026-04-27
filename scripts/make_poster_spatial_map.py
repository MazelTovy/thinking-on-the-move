"""Poster-ready version of `temporal_status_spatial_map.png`.

Differences vs `14_temporal_eval.py:plot_temporal_status_spatial_map`:
  - CartoDB Positron basemap tiles (subtle gray streets) beneath the points.
  - All status groups use circles (no triangle / star / square).
  - Stable dots are tiny + low-alpha so the long tail stops dominating.
  - New / Disappeared POIs are larger and opaque so they stand out.
  - No lat/lon axis ticks / labels — the basemap carries geographic context.
  - Size scales linearly with actual+predicted visit mass; no plateau jumps.

Output: experiments/<exp>/eval/figures/poster_temporal_status_spatial_map.{png,pdf}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/scratch/sx2490/econai/nyc_metro")
import importlib.util
spec = importlib.util.spec_from_file_location("t14", "/scratch/sx2490/econai/nyc_metro/14_temporal_eval.py")
t14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t14)


STATUS_COLOR = {
    "stable": "#9AA0A6",      # muted gray
    "disappeared": "#E68632",  # warm orange
    "new": "#D62728",          # strong red
    "unknown": "#9467BD",      # muted purple
}


def lonlat_to_mercator(lon, lat):
    """EPSG:4326 → EPSG:3857 Web Mercator (same as contextily tiles)."""
    R = 6378137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def size_linear(mass, mass_ref, base=1, cap=260):
    v = np.clip(np.asarray(mass, dtype=float) / max(mass_ref, 1e-12), 0, 1)
    return base + (cap - base) * v


def draw(actual, predicted, temporal_status_by_pid, poi_meta, out_png, out_pdf):
    records = t14.build_spatial_records(actual, predicted, temporal_status_by_pid)
    extent = t14.spatial_extent(records)
    if not records or extent is None:
        print("no records, aborting")
        return
    arr = t14.records_to_arrays(records, extent)
    total_mass = arr["actual"] + arr["predicted"]
    max_mass = max(float(total_mass.max()), 1e-12) if total_mass.size else 1e-12

    # Project lon/lat → Web Mercator for basemap alignment.
    xm, ym = lonlat_to_mercator(arr["lon"], arr["lat"])

    fig, ax = plt.subplots(figsize=(14, 13))

    # Draw least-interesting status first (stable), most-interesting last so
    # they sit on top. Stable uses very small, transparent dots to declutter.
    draw_order = [
        ("stable",      0.25, 1,   180),
        ("unknown",     0.55, 2,   220),
        ("disappeared", 0.85, 8,   480),
        ("new",         0.95, 10,  520),
    ]
    for status, alpha, base, cap in draw_order:
        mask = arr["status"] == status
        if not mask.any():
            continue
        sizes = size_linear(total_mass[mask], max_mass, base=base, cap=cap)
        ax.scatter(
            xm[mask], ym[mask], s=sizes, c=STATUS_COLOR[status], alpha=alpha,
            edgecolors="none",
            label=f"{status.capitalize()} ({int(mask.sum()):,})",
            rasterized=(status == "stable"),
            zorder={"stable": 2, "unknown": 3, "disappeared": 4, "new": 5}[status],
        )

    # Set extent with padding, then add basemap tiles underneath.
    pad_x = (xm.max() - xm.min()) * 0.04
    pad_y = (ym.max() - ym.min()) * 0.04
    ax.set_xlim(xm.min() - pad_x, xm.max() + pad_x)
    ax.set_ylim(ym.min() - pad_y, ym.max() + pad_y)

    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:3857",
                       source=cx.providers.CartoDB.PositronNoLabels,
                       zoom="auto", attribution_size=7)
    except Exception as e:
        print(f"[warn] basemap failed: {e}; continuing without tiles")

    # Hide axes / ticks / spines — basemap conveys geography.
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("2022 OOS POIs by Temporal Status",
                 fontsize=26, pad=14)
    leg = ax.legend(loc="upper right", fontsize=18, framealpha=0.93,
                    markerscale=2.0, labelspacing=0.7)
    leg.set_zorder(10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}\nwrote {out_pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="exp02_wcbg")
    p.add_argument("--out_stem", default="poster_temporal_status_spatial_map")
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
    valid_set, name_to_valid, poi_meta = t14.build_valid_poi_set(cbg_poi_dir, fallback)
    temporal_status_by_pid = t14.load_temporal_status_map(valid_set, name_to_valid)

    test_eval = t14.load_visit_level_eval(str(exp_dir / "pred_oos_2022.jsonl"),
                                          valid_set, name_to_valid)
    actual = test_eval["actual_top5"]; predicted = test_eval["predicted_top5"]

    draw(actual, predicted, temporal_status_by_pid, poi_meta,
         fig_dir / f"{args.out_stem}.png", fig_dir / f"{args.out_stem}.pdf")


if __name__ == "__main__":
    main()
