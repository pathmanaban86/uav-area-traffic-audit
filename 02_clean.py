# %% [markdown]
# # Cleaning — UAV microscopic vehicle trajectories (Delhi / Noida)
#
# Input: the per-site parquet files written by the EDA notebook (`EDA_results/standardised/`).
# Output: `<BASE>/cleaned/` — one cleaned parquet per site, a step-by-step cleaning log,
# figures, and `CLEANING_report.md`.
#
# Rules, all taken from the EDA findings:
# 1. Drop static tracks (no net movement) — mostly pedestrians and stopped objects.
# 2. Drop tracks shorter than 2 s.
# 3. Trim the first and last 1 s of every track — 97–99% of |acc| > 6 m/s² rows sit there.
# 4. Flag (do not delete) residual implausible rows: |lon. acc| > 6, |lat. acc| > 4 m/s², speed > 100 km/h.
# 5. Split each direction into lateral corridors (main carriageway, service lane, …) at empty gaps.
# 6. Flag tracks that start or end mid-road, and say whether those ends cluster (an access point)
#    or spread out (tracking loss).
# 5a. Flag U-turn tracks (heading change > 150°) and locate them; they are left out when finding corridors.
# 7. Flag rows whose footprint overlaps another vehicle in the same frame, direction and corridor,
#    and compare the files' footprints with typical class sizes.
# 8. Give every track a 10-minute block by its start time, for leakage-free train/test splits.
#
# Rules 4–7 only add flag columns; the modelling stage decides what to exclude.

# %% 0 · Setup
import os, glob, gc, json, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("EDA_HEADLESS"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

BASE = os.environ.get("EDA_BASE", "/content/drive/MyDrive/Research/SAGE_transport")
STD = os.path.join(BASE, "EDA_results", "standardised")
OUT = os.path.join(BASE, "cleaned")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
for _d in (OUT, FIG, TAB):
    os.makedirs(_d, exist_ok=True)
print("Reading :", STD)
print("Writing :", OUT)

# %% 1 · Configuration
MIN_TRACK_S = 2.0        # rule 2
END_TRIM_S = 1.0         # rule 3
ACC_LIMIT = 6.0          # rule 4, m/s²
LAT_ACC_LIMIT = 4.0      # rule 4, m/s²
SPEED_LIMIT = 100.0      # rule 4, km/h
BAND_BIN_M = 0.25        # rule 5: lateral histogram bin
BAND_GAP_M = 0.5         # rule 5: a near-empty strip at least this wide separates two corridors
BAND_EMPTY_FRAC = 0.02   # rule 5: "near-empty" = below 2% of the typical (median) occupied bin
BAND_MIN_SHARE = 0.01    # rule 5: corridors with fewer tracks than this share are labelled -1 (minor)
EDGE_BUFFER_M = 15.0     # rule 6 (trimming moves each end inward by up to ~15 m)
CLUSTER_WINDOW_M = 10.0  # rule 6: densest 10 m window of mid-road ends
CLUSTER_SHARE = 0.40     # rule 6: ≥ 40% of mid-road ends in that window → a physical access point
OVERLAP_RADIUS_M = 11.5  # rule 7: covers a 10.5 m bus beside any vehicle
BLOCK_S = 600            # rule 8
ONLY_SITES = []          # e.g. ["DEL_2"]

STABLE_LAT_RANGE_M = 2.0 # rule 5: corridors are found from tracks that drift < 2 m sideways
UTURN_HEADING_DEG = 150  # rule 5a: heading change marking a U-turn
UTURN_MIN_TRAVEL_M = 10  # rule 5a: ignore heading noise on tracks that barely move
FOOTPRINT_SOURCE = "file"   # rule 7: "file" (Track Length/Width) or "default" (typical class sizes)
FOOTPRINT_COMPARE_STEP = 5  # rule 7: compare both footprint sources on every 5th frame

MOTORISED = ["Car", "M2W", "M3W", "LCV", "Bus", "HCV"]
DEFAULT_DIMS = {"Car": (4.0, 1.7), "M2W": (1.9, 0.7), "M3W": (2.6, 1.3), "LCV": (5.0, 1.9),
                "Bus": (10.5, 2.5), "HCV": (8.5, 2.5), "Bicycle": (1.7, 0.6), "Pedestrian": (0.5, 0.5)}
NMT = ["Bicycle", "Pedestrian"]

# Paper Table 2 (speed km/h: mean, median, SD, range | acc m/s²: mean, median, SD, range)
PAPER_T2 = {
    "DEL_2": (33.53, 33.92, 6.92, 38.31, 0.059, 0.051, 0.299, 1.677),
    "DEL_3": (11.22, 9.72, 7.57, 33.30, -0.016, -0.006, 0.436, 2.339),
    "DEL_4": (34.93, 35.09, 8.43, 46.88, -0.0032, 0.0013, 0.3299, 1.822),
    "DEL_5": (36.08, 36.06, 9.64, 53.33, 0.1298, 0.1273, 0.2839, 1.5776),
    "NDA_1": (33.27, 33.93, 18.41, 88.48, -0.0007, 0.0062, 0.4088, 2.2374),
    "NDA_2": (28.09, 30.09, 17.10, 77.36, 0.0607, 0.0303, 0.4099, 2.2146),
}


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    if not os.environ.get("EDA_HEADLESS"):
        plt.show()
    plt.close(fig)


# %% 2 · Helpers
def find_corridor_boundaries(y):
    """Lateral positions of near-empty strips ≥ BAND_GAP_M wide (median, kerbs, separators)."""
    y = y[np.isfinite(y)]
    if len(y) < 100:
        return [], None, None
    lo, hi = np.percentile(y, [0.1, 99.9])
    bins = np.arange(lo, hi + BAND_BIN_M, BAND_BIN_M)
    h, _ = np.histogram(y, bins)
    hs = h.astype(float)
    ref = np.median(h[h > 0]) if (h > 0).any() else 0.0
    empty = hs < BAND_EMPTY_FRAC * ref
    bounds, run = [], 0
    for k, e in enumerate(empty):
        if e:
            run += 1
            continue
        if run * BAND_BIN_M >= BAND_GAP_M and k - run > 0:
            bounds.append(float(bins[k - run // 2]))
        run = 0
    return bounds, bins, hs


def densest_window_share(values, window):
    v = np.sort(values[np.isfinite(values)])
    if len(v) < 5:
        return np.nan, np.nan
    j = np.searchsorted(v, v + window, side="right")
    counts = j - np.arange(len(v))
    k = int(np.argmax(counts))
    return counts[k] / len(v), float(v[k] + window / 2)


def uturn_flags(df):
    """Per track: True if the travel heading turns by more than UTURN_HEADING_DEG; also where it happens."""
    vid = df["vid"].to_numpy(); x = df["xl"].to_numpy(float); y = df["yl"].to_numpy(float)
    starts = np.flatnonzero(np.r_[True, vid[1:] != vid[:-1]]); ends = np.r_[starts[1:], len(vid)]
    out = {}
    lim = np.radians(UTURN_HEADING_DEG)
    for s_, e in zip(starts, ends):
        xs, ys = x[s_:e:10], y[s_:e:10]            # about 1 s steps at 10 Hz
        if len(xs) < 3:
            continue
        dx, dy = np.diff(xs), np.diff(ys)
        mv = np.hypot(dx, dy) > 0.5
        if mv.sum() < 2 or np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) + np.abs(dx).sum() < UTURN_MIN_TRAVEL_M:
            continue
        h = np.unwrap(np.arctan2(dy[mv], dx[mv]))
        if h.max() - h.min() > lim:
            k = int(np.argmax(np.abs(h - h[0]) > np.pi / 2))
            out[vid[s_]] = float(xs[1:][mv][k])
    return out


def overlap_flags(df, Lv=None, Wv=None):
    """Axis-aligned footprint overlap within the same frame, direction and corridor."""
    n = len(df)
    flag = np.zeros(n, dtype=bool)
    fr = df["frame"].to_numpy(); dr = df["dir"].to_numpy(); co = df["corridor"].to_numpy()
    order = np.lexsort((co, dr, fr))
    fr, dr, co = fr[order], dr[order], co[order]
    S = df["s"].to_numpy()[order]; Y = df["yl"].to_numpy()[order]
    L = (df["L"].to_numpy() if Lv is None else np.asarray(Lv))[order]
    W = (df["W"].to_numpy() if Wv is None else np.asarray(Wv))[order]
    change = np.r_[True, (fr[1:] != fr[:-1]) | (dr[1:] != dr[:-1]) | (co[1:] != co[:-1])]
    starts = np.flatnonzero(change); ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        P = np.c_[S[s:e], Y[s:e]]
        pr = cKDTree(P).query_pairs(OVERLAP_RADIUS_M, output_type="ndarray")
        if not len(pr):
            continue
        i, j = pr[:, 0] + s, pr[:, 1] + s
        ov = (np.abs(S[i] - S[j]) < (L[i] + L[j]) / 2) & (np.abs(Y[i] - Y[j]) < (W[i] + W[j]) / 2)
        flag[order[i[ov]]] = True
        flag[order[j[ov]]] = True
    return flag


def stats(series):
    s = series.dropna()
    return dict(mean=s.mean(), median=s.median(), sd=s.std(), range=s.max() - s.min(),
                p01=s.quantile(0.01), p99=s.quantile(0.99))


# %% 3 · Clean one site
def clean_site(path):
    site = os.path.splitext(os.path.basename(path))[0]
    print(f"\n=== {site} ===")
    df = pd.read_parquet(path)
    df = df.sort_values(["vid", "t"], kind="mergesort").reset_index(drop=True)
    dt_nom = float(df.groupby("vid")["t"].diff().median())
    acc_before = stats(df["lon_acc"]) if "lon_acc" in df else {}
    spikes_before = int((df["lon_acc"].abs() > ACC_LIMIT).sum()) if "lon_acc" in df else 0
    log = []

    def note(step):
        log.append({"site": site, "step": step, "tracks": int(df["vid"].nunique()), "rows": int(len(df)),
                    "motorised_tracks": int(df.loc[df["cls"].isin(MOTORISED), "vid"].nunique())})
        print(f"  {step:<44s} tracks {log[-1]['tracks']:>6,}  rows {log[-1]['rows']:>10,}")

    note("0 · as loaded")
    df["is_nmt"] = df["cls"].isin(NMT)

    # rule 5a — U-turns (they cross the median and would hide its gap)
    ut = uturn_flags(df)
    df["flag_uturn"] = df["vid"].isin(ut.keys())
    ut_x = np.array(list(ut.values()))
    ut_share, ut_at = densest_window_share(ut_x, CLUSTER_WINDOW_M) if len(ut_x) >= 5 else (np.nan, np.nan)
    print(f"  U-turn tracks: {len(ut)}" + (f" · {ut_share:.0%} within 10 m of xl ≈ {ut_at:.0f} m" if ut_share == ut_share else ""))

    # rule 1 — static tracks (U-turners keep their track; their direction is set by how they start)
    if len(ut):
        first_moves = (df[df["flag_uturn"]].groupby("vid")["xl"]
                       .agg(lambda v: np.sign(v.iloc[min(len(v) - 1, 30)] - v.iloc[0])))
        fix = df["flag_uturn"] & (df["dir"] == 0)
        df.loc[fix, "dir"] = df.loc[fix, "vid"].map(first_moves).fillna(1).astype(df["dir"].dtype)
    df = df[(df["dir"] != 0) | df["flag_uturn"]]
    note("1 · drop static tracks")

    # rule 2 — short tracks
    g = df.groupby("vid")["t"]
    dur = g.transform("max") - g.transform("min")
    df = df[dur >= MIN_TRACK_S]
    note(f"2 · drop tracks < {MIN_TRACK_S:g} s")

    # rule 3 — trim track ends
    g = df.groupby("vid")["t"]
    t0, t1 = g.transform("min"), g.transform("max")
    df = df[((df["t"] - t0) >= END_TRIM_S) & ((t1 - df["t"]) >= END_TRIM_S)]
    g = df.groupby("vid")["t"]
    df = df[(g.transform("max") - g.transform("min")) > 0]
    note(f"3 · trim {END_TRIM_S:g} s at each track end")
    spikes_after = int((df["lon_acc"].abs() > ACC_LIMIT).sum()) if "lon_acc" in df else 0

    # rule 4 — flags for residual implausible values
    df["flag_acc"] = False
    if "lon_acc" in df:
        df["flag_acc"] |= df["lon_acc"].abs() > ACC_LIMIT
    if "lat_acc" in df:
        df["flag_acc"] |= df["lat_acc"].abs() > LAT_ACC_LIMIT
    df["flag_speed"] = df["speed"] > SPEED_LIMIT if "speed" in df else False

    # along-travel coordinate s: 0 at the upstream end for each direction
    df["s"] = df["xl"] * df["dir"]
    for d in (1, -1):
        m = df["dir"] == d
        if m.any():
            df.loc[m, "s"] -= np.nanpercentile(df.loc[m, "s"], 0.5)

    # rule 5 — lateral corridors per direction (boundaries from stable, motorised through-traffic)
    df["corridor"] = -1
    corridor_rows = []
    trk = df.groupby("vid").agg(dir=("dir", "first"), cls=("cls", "first"), y_med=("yl", "median"),
                                y_min=("yl", "min"), y_max=("yl", "max"))
    trk["corridor"] = -1
    trk["flag_cross_corridor"] = False
    stable = trk.index[((trk["y_max"] - trk["y_min"]) < STABLE_LAT_RANGE_M) & ~trk.index.isin(ut.keys())
                       & ~trk["cls"].isin(NMT)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.6))
    for a_, d in zip(axs, (1, -1)):
        rows_d = (df["dir"] == d) & ~df["is_nmt"]
        st_rows = rows_d & df["vid"].isin(stable)
        ymot = df.loc[st_rows if df.loc[st_rows, "vid"].nunique() >= 200 else rows_d, "yl"].to_numpy()
        bounds, bins, hs = find_corridor_boundaries(ymot)
        sel = trk["dir"] == d
        if not sel.any():
            a_.set_axis_off(); continue
        band = np.searchsorted(bounds, trk.loc[sel, "y_med"].to_numpy())
        share = pd.Series(band).value_counts(normalize=True)
        keep = set(share[share >= BAND_MIN_SHARE].index)
        trk.loc[sel, "corridor"] = [b if b in keep else -1 for b in band]
        if bounds:
            lo_b = np.searchsorted(bounds, trk.loc[sel, "y_min"].to_numpy())
            hi_b = np.searchsorted(bounds, trk.loc[sel, "y_max"].to_numpy())
            trk.loc[sel, "flag_cross_corridor"] = lo_b != hi_b
        if bins is not None:
            a_.fill_between(bins[:-1], hs, step="post", color="#3b6ea5")
            for b in bounds:
                a_.axvline(b, color="#c0504d", ls="--")
            a_.set_yscale("log"); a_.set_xlabel("lateral position yl (m)"); a_.set_ylabel("rows (log)")
        a_.set_title(f"{'main' if d == 1 else 'opposite'} direction · {len(bounds)} boundaries")
        for b in sorted(keep):
            ys = df.loc[(df["dir"] == d) & df["vid"].isin(trk.index[sel & (trk["corridor"] == b)]), "yl"]
            corridor_rows.append({"site": site, "dir": d, "corridor": int(b),
                                  "tracks": int((sel & (trk["corridor"] == b)).sum()),
                                  "y_from_m": round(float(np.nanpercentile(ys, 0.5)), 2),
                                  "y_to_m": round(float(np.nanpercentile(ys, 99.5)), 2),
                                  "width_m": round(float(np.nanpercentile(ys, 99.5) - np.nanpercentile(ys, 0.5)), 2)})
    fig.suptitle(f"{site} · lateral corridors (red lines = near-empty strips ≥ {BAND_GAP_M:g} m)", y=1.04)
    savefig(fig, f"{site}_corridors.png")
    df["corridor"] = df["vid"].map(trk["corridor"]).astype(np.int16)
    df["flag_cross_corridor"] = df["vid"].map(trk["flag_cross_corridor"]).astype(bool)
    corr_tab = pd.DataFrame(corridor_rows)
    print(corr_tab.to_string(index=False) if len(corr_tab) else "  no corridors found")

    # rule 6 — mid-road starts and ends, per direction and corridor
    tg = df.groupby("vid")
    tt = pd.DataFrame({"dir": tg["dir"].first(), "corridor": tg["corridor"].first(), "cls": tg["cls"].first(),
                       "t0": tg["t"].first(), "t1": tg["t"].last(),
                       "s_entry": tg["s"].first(), "s_exit": tg["s"].last()})
    tt["starts_mid_road"] = False; tt["ends_mid_road"] = False
    tmin, tmax = df["t"].min() + 2, df["t"].max() - 2
    mid_rows = []
    for (d, c), sub in tt.groupby(["dir", "corridor"]):
        if c < 0 or len(sub) < 20:
            continue
        rows = df["dir"].eq(d) & df["corridor"].eq(c)
        lo, hi = np.nanpercentile(df.loc[rows, "s"], [1, 99])
        st = (sub["s_entry"] > lo + EDGE_BUFFER_M) & (sub["t0"] > tmin)
        en = (sub["s_exit"] < hi - EDGE_BUFFER_M) & (sub["t1"] < tmax)
        tt.loc[sub.index, "starts_mid_road"] = st
        tt.loc[sub.index, "ends_mid_road"] = en
        sh_s, at_s = densest_window_share(sub.loc[st, "s_entry"].to_numpy(), CLUSTER_WINDOW_M)
        sh_e, at_e = densest_window_share(sub.loc[en, "s_exit"].to_numpy(), CLUSTER_WINDOW_M)
        mid_rows.append({"site": site, "dir": d, "corridor": c, "tracks": len(sub),
                         "starts_mid_%": round(100 * st.mean(), 1), "ends_mid_%": round(100 * en.mean(), 1),
                         "starts_densest10m_share": round(sh_s, 2) if sh_s == sh_s else None,
                         "starts_cluster_at_s_m": round(at_s, 1) if at_s == at_s else None,
                         "ends_densest10m_share": round(sh_e, 2) if sh_e == sh_e else None,
                         "ends_cluster_at_s_m": round(at_e, 1) if at_e == at_e else None,
                         "ends_reading": ("clustered — access point / turn" if sh_e >= CLUSTER_SHARE
                                          else "spread — tracking loss") if sh_e == sh_e else "too few"})
    mid_tab = pd.DataFrame(mid_rows)
    if len(mid_tab):
        print(mid_tab.to_string(index=False))
    df["starts_mid_road"] = df["vid"].map(tt["starts_mid_road"]).astype(bool)
    df["ends_mid_road"] = df["vid"].map(tt["ends_mid_road"]).astype(bool)

    main = tt[(tt["dir"] == 1) & (tt["corridor"] >= 0)]
    if len(main):
        fig, axs = plt.subplots(1, 2, figsize=(14, 3.6))
        axs[0].hist(main.loc[main["starts_mid_road"], "s_entry"], bins=60, color="#4f81bd")
        axs[0].set_xlabel("s at first point (m)"); axs[0].set_title("mid-road starts, main direction")
        axs[1].hist(main.loc[main["ends_mid_road"], "s_exit"], bins=60, color="#c0504d")
        axs[1].set_xlabel("s at last point (m)"); axs[1].set_title("mid-road ends, main direction")
        fig.suptitle(f"{site} · where tracks appear and disappear (peaks = access points, flat = tracking loss)")
        savefig(fig, f"{site}_mid_road_positions.png")

    # rule 7 — footprint overlaps: compare file footprints with typical class sizes, then flag
    Ldef = df["cls"].map(lambda c: DEFAULT_DIMS.get(c, (4.0, 1.7))[0]).to_numpy()
    Wdef = df["cls"].map(lambda c: DEFAULT_DIMS.get(c, (4.0, 1.7))[1]).to_numpy()
    samp = (df["frame"] % FOOTPRINT_COMPARE_STEP == 0).to_numpy()
    sub = df.loc[samp]
    ov_file = float(overlap_flags(sub).mean())
    ov_def = float(overlap_flags(sub, Ldef[samp], Wdef[samp]).mean())
    fp = (df.groupby("vid").agg(cls=("cls", "first"), L=("L", "median"), W=("W", "median"))
            .groupby("cls")[["L", "W"]].median().round(2))
    fp["L_typical"] = [DEFAULT_DIMS.get(c, (np.nan, np.nan))[0] for c in fp.index]
    fp["W_typical"] = [DEFAULT_DIMS.get(c, (np.nan, np.nan))[1] for c in fp.index]
    fp["L_ratio"] = (fp["L"] / fp["L_typical"]).round(2); fp["W_ratio"] = (fp["W"] / fp["W_typical"]).round(2)
    fp.index.name = "class"
    fp.reset_index().assign(site=site).to_csv(os.path.join(TAB, f"{site}_footprints.csv"), index=False)
    print(fp.to_string())
    print(f"  rows in an overlap (every {FOOTPRINT_COMPARE_STEP}th frame): file footprints {ov_file:.1%} · "
          f"typical class sizes {ov_def:.1%}")
    if FOOTPRINT_SOURCE == "default":
        df["flag_overlap"] = overlap_flags(df, Ldef, Wdef)
    else:
        df["flag_overlap"] = overlap_flags(df)

    # rule 8 — time blocks by track start
    df["block"] = (df["vid"].map(tt["t0"]) // BLOCK_S).astype(np.int16)

    note("final (rules 4–8 add flags only)")

    # ---- acceleration before vs after trimming
    acc_after = stats(df["lon_acc"]) if "lon_acc" in df else {}
    if "lon_acc" in df:
        fig, a_ = plt.subplots(figsize=(8, 3.6))
        raw = pd.read_parquet(path, columns=["lon_acc"])["lon_acc"].dropna()
        bins = np.linspace(-15, 8, 120)
        a_.hist(raw.clip(-15, 8), bins=bins, alpha=0.6, label="as released", color="#7f7f7f")
        a_.hist(df["lon_acc"].dropna().clip(-15, 8), bins=bins, alpha=0.8, label="after cleaning", color="#3b6ea5")
        a_.set_yscale("log"); a_.set_xlabel("longitudinal acceleration (m/s², clipped to −15…8)"); a_.legend()
        a_.set_title(f"{site} · effect of trimming track ends")
        savefig(fig, f"{site}_acc_before_after.png")
        del raw

    sp = stats(df.loc[~df["is_nmt"], "speed"])
    flags = {k: round(100 * float(df[k].mean()), 3) for k in
             ["flag_acc", "flag_speed", "flag_cross_corridor", "flag_uturn", "flag_overlap",
              "starts_mid_road", "ends_mid_road"]}

    summary = {
        "site": site, "sampling_hz": round(1 / dt_nom, 2),
        "tracks_in": log[0]["tracks"], "tracks_out": log[-1]["tracks"],
        "rows_in": log[0]["rows"], "rows_out": log[-1]["rows"],
        "rows_kept_%": round(100 * log[-1]["rows"] / log[0]["rows"], 1),
        "acc_spikes_before": spikes_before, "acc_spikes_after": spikes_after,
        "acc_sd_before": round(acc_before.get("sd", np.nan), 3), "acc_sd_after": round(acc_after.get("sd", np.nan), 3),
        "acc_p01_after": round(acc_after.get("p01", np.nan), 2), "acc_p99_after": round(acc_after.get("p99", np.nan), 2),
        "speed_mean_after": round(sp["mean"], 2), "speed_sd_after": round(sp["sd"], 2),
        "corridors": len(corr_tab), "uturn_tracks": len(ut),
        "uturn_densest10m_share": None if ut_share != ut_share else round(ut_share, 2),
        "uturn_at_xl_m": None if ut_at != ut_at else round(ut_at, 1),
        "overlap_rows_%_file_dims": round(100 * ov_file, 2), "overlap_rows_%_typical_dims": round(100 * ov_def, 2),
        "footprint_L_ratio_car": fp["L_ratio"].get("Car", np.nan), "footprint_W_ratio_car": fp["W_ratio"].get("Car", np.nan),
        "footprint_W_ratio_m2w": fp["W_ratio"].get("M2W", np.nan),
        **{f"{k}_%": v for k, v in flags.items()},
    }
    if site in PAPER_T2:
        r = PAPER_T2[site]
        summary.update(paper_speed_mean=r[0], paper_speed_sd=r[2], paper_acc_sd=r[6])

    keep_cols = [c for c in ["vid", "dir", "corridor", "block", "frame", "t", "x", "y", "xl", "yl", "s", "speed",
                             "lon_acc", "lat_acc", "cls", "vtype_raw", "is_nmt", "L", "W",
                             "flag_acc", "flag_speed", "flag_cross_corridor", "flag_uturn", "flag_overlap",
                             "starts_mid_road", "ends_mid_road"] if c in df]
    df[keep_cols].to_parquet(os.path.join(OUT, f"{site}_clean.parquet"), index=False)
    pd.DataFrame(log).to_csv(os.path.join(TAB, f"{site}_cleaning_log.csv"), index=False)
    if len(corr_tab):
        corr_tab.to_csv(os.path.join(TAB, f"{site}_corridors.csv"), index=False)
    if len(mid_tab):
        mid_tab.to_csv(os.path.join(TAB, f"{site}_mid_road.csv"), index=False)
    return summary, pd.DataFrame(log), corr_tab, mid_tab


# %% 4 · Run all sites
paths = sorted(glob.glob(os.path.join(STD, "*.parquet")), key=os.path.getsize)
if ONLY_SITES:
    paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in ONLY_SITES]
print("Sites:", [os.path.splitext(os.path.basename(p))[0] for p in paths])
SUMS, LOGS, CORRS, MIDS = [], [], [], []
for p in paths:
    try:
        s, l, c, m = clean_site(p)
        SUMS.append(s); LOGS.append(l); CORRS.append(c); MIDS.append(m)
    except MemoryError:
        print(f"  out of memory on {p} — use a High-RAM runtime or ONLY_SITES")
    gc.collect()

SUM = pd.DataFrame(SUMS)
SUM.to_csv(os.path.join(TAB, "ALL_cleaning_summary.csv"), index=False)
print(SUM.to_string(index=False))

# %% 5 · Report
def md(d):
    try:
        return d.to_markdown(index=False)
    except Exception:
        return "```\n" + d.to_string(index=False) + "\n```"


cat = lambda L_: pd.concat([x for x in L_ if len(x)], ignore_index=True) if any(len(x) for x in L_) else pd.DataFrame()
lines = [f"# Cleaning report ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"Rules: drop static tracks; drop tracks < {MIN_TRACK_S:g} s; trim {END_TRIM_S:g} s at each end; "
         f"flag |lon. acc| > {ACC_LIMIT:g}, |lat. acc| > {LAT_ACC_LIMIT:g} m/s², speed > {SPEED_LIMIT:g} km/h; "
         f"corridors split at near-empty strips ≥ {BAND_GAP_M:g} m; footprint-overlap and mid-road flags; "
         f"{BLOCK_S // 60}-min blocks by track start.", "",
         "## Summary", "", md(SUM) if len(SUM) else "_none_", "",
         "## Step-by-step log", "", md(cat(LOGS)), "",
         "## Lateral corridors", "", md(cat(CORRS)), "",
         "## Mid-road starts and ends", "", md(cat(MIDS)), "",
         "## Figures", ""]
lines += [f"![{f}](figures/{f})" for f in sorted(os.listdir(FIG))]
with open(os.path.join(OUT, "CLEANING_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nDone:", os.path.join(OUT, "CLEANING_report.md"))
