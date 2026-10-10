# %% [markdown]
# # Exploratory data analysis — UAV microscopic vehicle trajectories (Delhi / Noida, area-based traffic)
#
# Dataset: Ali, Rao, Bhaskar & Chatterjee (2025), arXiv 2512.11898, Zenodo 10.5281/zenodo.17745347.
#
# Purpose: audit the data **before any modelling** — schema, sampling, track integrity, units,
# kinematic consistency, measurement noise, footprint overlaps, class composition, and agreement
# with the paper's Table 1 / Table 2. Nothing in the raw files is modified.
#
# Run top to bottom in Google Colab. All outputs are written to `<BASE>/EDA_results/`:
# `tables/` (CSV), `figures/` (PNG), `standardised/` (parquet, one per site),
# `EDA_report.md` and `eda_flags.json`.

# %% 0 · Setup
import os, re, json, glob, gc, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("EDA_HEADLESS"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

BASE = os.environ.get("EDA_BASE", "/content/drive/MyDrive/Research/SAGE_transport")
OUT = os.path.join(BASE, "EDA_results")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
STD = os.path.join(OUT, "standardised")
for _d in (OUT, FIG, TAB, STD):
    os.makedirs(_d, exist_ok=True)
print("Input folder :", BASE)
print("Output folder:", OUT)

# %% 1 · Configuration (edit here if auto-detection gets something wrong)
FPS = 30                 # frame rate stated in the paper; the real sampling interval is detected per file
SG_WIN_S = 0.5           # s, Savitzky–Golay window for positions/speed (converted to samples per file)
SG_POLY = 3
SG_WIN_ACC_S = 1.0       # s, longer window for acceleration, which amplifies position noise
INTERACTION_STEP_S = 0.5 # s, pairwise checks every 0.5 s to keep DEL_2/DEL_3 tractable
SWITCH_DT_S = 1.0        # s, ID-switch test: a new track starting within this time ...
SWITCH_DX_M = 4.0        # m, ... and within this distance of where a mid-road track ended
END_WINDOW_S = 1.0       # s, first/last part of a track examined for acceleration spikes
NEIGHBOUR_RADIUS = 25.0  # m, search radius for leader / side neighbour
EDGE_BUFFER_M = 10.0     # m, tracks that start/end further than this inside the stretch = fragments
FD_WINDOW_S = 30         # s, Edie aggregation window for fundamental diagrams
TS_WINDOW_S = 300        # s, length of the time–space diagram excerpt
ONLY_SITES = []          # e.g. ["DEL_2"] to (re)run one site if Colab runs out of RAM; [] = all

# Force a raw column for a role, e.g. {"vid": "Track ID", "time": "Time [s]"}; per file if needed:
COLUMN_OVERRIDE = {}               # applies to all files
COLUMN_OVERRIDE_PER_SITE = {}      # e.g. {"DEL_3": {"lon_acc": "Tan. Acc. [ms-2]"}}

# Raw class label (normalised: lowercase, alphanumerics only) -> canonical class used in the paper
CLASS_MAP = {
    "car": "Car", "taxi": "Car", "passengercar": "Car",
    "m2w": "M2W", "mtw": "M2W", "2w": "M2W", "twowheeler": "M2W", "motorcycle": "M2W",
    "motorbike": "M2W", "scooter": "M2W", "bike": "M2W",
    "m3w": "M3W", "mthw": "M3W", "3w": "M3W", "threewheeler": "M3W", "auto": "M3W",
    "autorickshaw": "M3W", "rickshaw": "M3W", "tricycle": "M3W",
    "lcv": "LCV", "mediumvehicle": "LCV", "van": "LCV", "lightcommercialvehicle": "LCV",
    "bus": "Bus",
    "hcv": "HCV", "heavyvehicle": "HCV", "truck": "HCV", "heavycommercialvehicle": "HCV",
    "tuktuk": "M3W",                                    # Data-from-Sky's label for auto-rickshaws
    "bicycle": "Bicycle", "cycle": "Bicycle", "pedestrian": "Pedestrian", "person": "Pedestrian",
}
CLASS_ORDER = ["Car", "M2W", "M3W", "LCV", "Bus", "HCV"]   # motorised classes used in the paper
NMT_CLASSES = ["Bicycle", "Pedestrian"]                   # kept, but excluded from paper comparisons
# Typical footprints (length, width in m) — used ONLY if the file has no length/width columns
DEFAULT_DIMS = {"Car": (4.0, 1.7), "M2W": (1.9, 0.7), "M3W": (2.6, 1.3),
                "LCV": (5.0, 1.9), "Bus": (10.5, 2.5), "HCV": (8.5, 2.5),
                "Bicycle": (1.7, 0.6), "Pedestrian": (0.5, 0.5)}

# ---- Reference values transcribed from the paper -------------------------------------
# Table 1: duration (min), composition % Car:M2W:M3W:LCV:Bus:HCV, stretch (m), width (m)
PAPER_T1 = {
    "DEL_2": (63, [41, 34, 15, 8, 1, 1], 180, 14.0),
    "DEL_3": (63, [40, 31, 15, 11, 2, 1], 180, 14.0),
    "DEL_4": (66, [41, 40, 11, 5, 1, 1], 200, 10.5),
    "DEL_5": (66, [35, 45, 11, 6, 1, 2], 200, 10.5),
    "NDA_1": (41, [41, 39, 10, 4, 1, 5], 170, 14.0),
    "NDA_2": (41, [38, 39, 11, 5, 1, 6], 170, 14.0),
}
# Table 2: speed (km/h) mean, median, SD, range | acceleration (m/s2) mean, median, SD, range
PAPER_T2 = {
    "DEL_2": (33.53, 33.92, 6.92, 38.31, 0.059, 0.051, 0.299, 1.677),
    "DEL_3": (11.22, 9.72, 7.57, 33.30, -0.016, -0.006, 0.436, 2.339),
    "DEL_4": (34.93, 35.09, 8.43, 46.88, -0.0032, 0.0013, 0.3299, 1.822),
    "DEL_5": (36.08, 36.06, 9.64, 53.33, 0.1298, 0.1273, 0.2839, 1.5776),
    "NDA_1": (33.27, 33.93, 18.41, 88.48, -0.0007, 0.0062, 0.4088, 2.2374),
    "NDA_2": (28.09, 30.09, 17.10, 77.36, 0.0607, 0.0303, 0.4099, 2.2146),
}
EXPECTED_SITES = list(PAPER_T1)

FLAGS = []  # (site, severity, check, message)


def flag(site, severity, check, msg):
    FLAGS.append({"site": site, "severity": severity, "check": check, "message": msg})
    print(f"  [{severity}] {site} · {check}: {msg}")


# %% 2 · Loading helpers (handles long CSV and Data-from-Sky "wide" CSV)
def norm(s):
    s = re.sub(r"\[.*?\]|\(.*?\)", "", str(s))
    return re.sub(r"[^a-z0-9]", "", s.lower())


ROLE_CANDIDATES = {
    "vid":     ["vehicleid", "trackid", "id", "vehid", "objectid", "track", "vehicle", "vehicleno"],
    "time":    ["timestamp", "time", "times", "t", "frame", "frameid", "frameno", "framenumber"],
    "x":       ["x", "localx", "xcoordinate", "posx", "xpos", "xm", "xposition"],
    "y":       ["y", "localy", "ycoordinate", "posy", "ypos", "ym", "yposition"],
    "speed":   ["speed", "speedkmh", "velocity", "v", "spd", "vehiclespeed"],
    "lon_acc": ["longitudinalacceleration", "lonacc", "longacc", "tanacc", "tanaccel",
                "tangentialacceleration", "longitudinalacc", "acceleration", "acc", "accel"],
    "lat_acc": ["lateralacceleration", "latacc", "lataccel", "lateralacc"],
    "vtype":   ["vehicletype", "type", "class", "vehicleclass", "category", "vtype", "vehiclecategory"],
    "length":  ["length", "vehiclelength", "len", "l"],
    "width":   ["width", "vehiclewidth", "wid", "w"],
}


def detect_columns(columns):
    ncols = {c: norm(c) for c in columns}
    used, mapping = set(), {}
    for role, cands in ROLE_CANDIDATES.items():          # pass 1: exact
        for cand in cands:
            hit = [c for c, n in ncols.items() if n == cand and c not in used]
            if hit:
                mapping[role] = hit[0]; used.add(hit[0]); break
    for role, cands in ROLE_CANDIDATES.items():          # pass 2: substring, longer names only
        if role in mapping:
            continue
        for cand in cands:
            if len(cand) < 4:
                continue
            hit = [c for c, n in ncols.items() if cand in n and c not in used
                   and not n.startswith(("avg", "mean", "max", "min"))]
            if hit:
                mapping[role] = hit[0]; used.add(hit[0]); break
    return mapping


def sniff(path, nlines=6):
    with open(path, "r", errors="replace") as f:
        head = [f.readline() for _ in range(nlines)]
    first = head[0]
    delim = max([",", ";", "\t"], key=lambda d: first.count(d))
    hdr = [h.strip() for h in first.rstrip("\r\n").split(delim)]
    while hdr and hdr[-1] == "":
        hdr.pop()
    lens = [len([p for p in l.rstrip("\r\n").split(delim) if p.strip() != ""])
            for l in head[1:] if l.strip()]
    wide = bool(lens) and max(lens) > len(hdr) + 3
    return delim, hdr, wide


def load_wide(path, delim, hdr):
    """Data-from-Sky export: one row per vehicle, fixed fields then repeating groups."""
    nh = [norm(h) for h in hdr]
    start = next((i for i, h in enumerate(nh) if h in ("x", "xm", "localx", "lat")), None)
    if start is None:
        raise ValueError(f"Wide format detected but no x/lat column in header: {hdr}")
    fixed, group = hdr[:start], hdr[start:]
    g = len(group)
    print(f"  wide format: fixed={fixed} | repeating group ({g})={group}")
    parts_all = []
    with open(path, "r", errors="replace") as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.rstrip("\r\n").split(delim)]
            while parts and parts[-1] == "":
                parts.pop()
            if len(parts) <= start:
                continue
            vals = pd.to_numeric(pd.Series(parts[start:]), errors="coerce").to_numpy()
            n = len(vals) // g
            if n == 0:
                continue
            block = pd.DataFrame(vals[: n * g].reshape(n, g), columns=group)
            for c, v in zip(fixed, parts[:start]):
                block[c] = v
            parts_all.append(block)
    out = pd.concat(parts_all, ignore_index=True)
    for c in fixed:
        conv = pd.to_numeric(out[c], errors="coerce")
        if conv.notna().mean() > 0.99:
            out[c] = conv
    return out


def load_raw(path):
    delim, hdr, wide = sniff(path)
    print(f"  delimiter={delim!r}  wide={wide}  header={hdr[:12]}{' ...' if len(hdr) > 12 else ''}")
    if wide:
        return load_wide(path, delim, hdr), "wide (Data-from-Sky style)"
    raw = pd.read_csv(path, sep=delim, low_memory=False)
    raw = raw.loc[:, [c for c in raw.columns if not (str(c).startswith("Unnamed") and raw[c].isna().all())]]
    return raw, "long (one row per vehicle-frame)"


def site_name(fname):
    m = re.search(r"(DEL|NDA)[ _\-]?(\d)", fname.upper())
    return f"{m.group(1)}_{m.group(2)}" if m else os.path.splitext(fname)[0]


def canonical_class(v):
    return CLASS_MAP.get(norm(v), str(v).strip().title() if pd.notna(v) else "Unknown")


# %% 3 · Standardisation (rename, units, road-aligned axes) — raw values are kept alongside
def standardise(raw, site):
    mapping = detect_columns(raw.columns)
    mapping.update(COLUMN_OVERRIDE)
    mapping.update(COLUMN_OVERRIDE_PER_SITE.get(site, {}))
    print("  column mapping:", mapping)
    unmapped = [c for c in raw.columns if c not in mapping.values()]
    if unmapped:
        print("  unmapped raw columns:", unmapped)
    miss = [k for k in ("vid", "time", "x", "y") if k not in mapping]
    if miss:
        raise ValueError(f"{site}: cannot find {miss}. Set COLUMN_OVERRIDE_PER_SITE['{site}'].")
    for k in ("speed", "lon_acc", "lat_acc", "vtype", "length", "width"):
        if k not in mapping:
            flag(site, "WARN", "schema", f"no column found for '{k}'")

    df = pd.DataFrame({k: raw[c].values for k, c in mapping.items()})
    meta = {"mapping": mapping}

    # time → seconds
    if not np.issubdtype(df["time"].dtype, np.number):
        tt = pd.to_numeric(df["time"], errors="coerce")
        if tt.notna().mean() < 0.9:
            ts = pd.to_datetime(df["time"], errors="coerce")
            tt = (ts - ts.min()).dt.total_seconds()
        df["time"] = tt
    for k in ("x", "y", "speed", "lon_acc", "lat_acc", "length", "width"):
        if k in df:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    df = df.sort_values(["vid", "time"], kind="mergesort").reset_index(drop=True)
    med_dt = df.groupby("vid")["time"].diff().median()
    if 0.02 <= med_dt <= 0.25:
        unit, scale = "s", 1.0
    elif 0.5 <= med_dt <= 1.5:
        unit, scale = "frames", 1.0 / FPS
    elif 20 <= med_dt <= 250:
        unit, scale = "ms", 1e-3
    else:
        unit, scale = f"unknown (median step {med_dt:.4g})", 1.0
        flag(site, "HIGH", "time", f"median time step {med_dt:.4g} cannot be interpreted")
    df["t"] = (df["time"] - df["time"].min()) * scale
    dt_nom = float(med_dt * scale)
    df["frame"] = np.round(df["t"] / dt_nom).astype(np.int64)
    meta.update(time_unit=unit, dt_nom=dt_nom, sampling_hz=round(1 / dt_nom, 3))
    print(f"  time column unit: {unit} · sampling interval {dt_nom:.4f} s ({1 / dt_nom:.2f} Hz)")
    if abs(1 / dt_nom - FPS) / FPS > 0.1:
        flag(site, "INFO", "sampling", f"trajectories are sampled at {1 / dt_nom:.2f} Hz, not the {FPS} fps "
             "stated in the paper; smoothing windows and step sizes are set in seconds")

    # classes
    if "vtype" in df:
        df["vtype_raw"] = df["vtype"].astype(str)
        df["cls"] = df["vtype"].map(canonical_class)
    else:
        df["cls"] = "Unknown"
    labels = df.groupby("cls")["vid"].nunique().to_dict()
    print("  canonical classes (unique vehicles):", labels)
    odd = [c for c in labels if c not in CLASS_ORDER + NMT_CLASSES]
    if odd:
        flag(site, "WARN", "classes", f"labels not mapped to paper classes: {odd} — extend CLASS_MAP")

    # footprints
    for k, idx in (("length", 0), ("width", 1)):
        if k not in df or df[k].isna().all():
            df[k] = df["cls"].map(lambda c: DEFAULT_DIMS.get(c, (4.0, 1.7))[idx])
            meta[f"{k}_source"] = "class default"
        else:
            meta[f"{k}_source"] = "file"
    df["L"] = df["length"].fillna(df["cls"].map(lambda c: DEFAULT_DIMS.get(c, (4.0, 1.7))[0]))
    df["W"] = df["width"].fillna(df["cls"].map(lambda c: DEFAULT_DIMS.get(c, (4.0, 1.7))[1]))

    # road-aligned axes: PCA of positions, travel direction → +xl, 0 at the 0.5th percentile
    P = df[["x", "y"]].dropna().to_numpy()
    samp = P[np.random.default_rng(0).choice(len(P), min(len(P), 200_000), replace=False)]
    c0 = samp.mean(0)
    _, _, vt = np.linalg.svd(samp - c0, full_matrices=False)
    ax = vt[0]
    ang = np.degrees(np.arctan2(ax[1], ax[0]))
    xl = (df["x"] - c0[0]) * ax[0] + (df["y"] - c0[1]) * ax[1]
    yl = -(df["x"] - c0[0]) * ax[1] + (df["y"] - c0[1]) * ax[0]
    first = df.groupby("vid").head(1).index
    last = df.groupby("vid").tail(1).index
    net = xl.loc[last].to_numpy() - xl.loc[first].to_numpy()
    moving = np.abs(net) > 5
    sign = 1.0 if (net[moving] >= 0).mean() >= 0.5 else -1.0
    xl, yl = xl * sign, yl * sign
    df["xl"] = xl - np.nanpercentile(xl, 0.5)
    df["yl"] = yl - np.nanpercentile(yl, 0.5)
    wrong_way = float(((net[moving] * sign) < 0).mean()) if moving.any() else np.nan
    vids_first = df.loc[first, "vid"].to_numpy()
    direction = np.where(net * sign > 5, 1, np.where(net * sign < -5, -1, 0))
    df["dir"] = df["vid"].map(dict(zip(vids_first, direction))).astype(np.int8)
    meta.update(axis_angle_deg=round(float(ang), 2), wrong_way_share=wrong_way)
    print(f"  road axis at {ang:.1f}° to raw x; vehicles against main direction: {wrong_way:.2%}")
    df["v_ms"] = df["speed"] / 3.6 if "speed" in df else np.nan
    return df, meta


# %% 4 · Kinematic derivation per track (for consistency checks only)
def track_bounds(v):
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    return starts, np.r_[starts[1:], len(v)]


def odd_window(seconds, dt_nom, minimum=5):
    n = int(round(seconds / dt_nom))
    n = n if n % 2 else n + 1
    return max(minimum, n)


def derive_kinematics(df, dt_nom):
    w = odd_window(SG_WIN_S, dt_nom)
    wa = odd_window(SG_WIN_ACC_S, dt_nom)
    poly = min(SG_POLY, w - 2)
    print(f"  smoothing windows: {w} samples for position/speed, {wa} for acceleration")
    n = len(df)
    t = df["t"].to_numpy(float); x = df["xl"].to_numpy(float); y = df["yl"].to_numpy(float)
    a_rep = df["lon_acc"].to_numpy(float) if "lon_acc" in df else np.full(n, np.nan)
    keys = ["dt", "v_raw", "v_sg", "vx_sg", "vy_sg", "a_sg", "alat_sg", "jerk_rep", "res_x", "res_y"]
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in keys}
    starts, ends = track_bounds(df["vid"].to_numpy())
    for s, e in zip(starts, ends):
        m = e - s
        if m < 2:
            continue
        ts, xs, ys = t[s:e], x[s:e], y[s:e]
        dt = np.diff(ts)
        ok = dt > 0
        out["dt"][s + 1:e] = dt
        with np.errstate(divide="ignore", invalid="ignore"):
            out["v_raw"][s + 1:e] = np.where(ok, np.hypot(np.diff(xs), np.diff(ys)) / dt * 3.6, np.nan)
            out["jerk_rep"][s + 1:e] = np.where(ok, np.diff(a_rep[s:e]) / dt, np.nan)
        if m >= w and ok.all():
            h = float(np.median(dt))
            xs_s = savgol_filter(xs, w, poly); ys_s = savgol_filter(ys, w, poly)
            vx = savgol_filter(xs, w, poly, deriv=1, delta=h)
            vy = savgol_filter(ys, w, poly, deriv=1, delta=h)
            sp = np.hypot(vx, vy)
            out["v_sg"][s:e] = sp * 3.6
            out["vx_sg"][s:e] = vx * 3.6
            out["vy_sg"][s:e] = vy * 3.6
            if m >= wa:
                out["a_sg"][s:e] = savgol_filter(sp, wa, 2, deriv=1, delta=h)
                out["alat_sg"][s:e] = savgol_filter(ys, wa, 2, deriv=2, delta=h)
            out["res_x"][s:e] = xs - xs_s
            out["res_y"][s:e] = ys - ys_s
    for k in keys:
        df[k] = out[k]
    return df


def track_table(df):
    g = df.groupby("vid", sort=False)
    tt = pd.DataFrame({
        "cls": g["cls"].first(), "dir": g["dir"].first(), "n_points": g.size(),
        "t0": g["t"].first(), "t1": g["t"].last(),
        "x_entry": g["xl"].first(), "x_exit": g["xl"].last(),
        "y_entry": g["yl"].first(), "y_exit": g["yl"].last(),
        "y_mean": g["yl"].mean(), "dt_max": g["dt"].max(),
        "L_med": g["L"].median(), "W_med": g["W"].median(),
        "L_cv": g["L"].std() / g["L"].mean(), "W_cv": g["W"].std() / g["W"].mean(),
        "n_classes": g["cls"].nunique(),
    })
    vcol = "speed" if "speed" in df else "v_sg"
    tt["v_mean"] = g[vcol].mean(); tt["v_max"] = g[vcol].max(); tt["v_end"] = g[vcol].last()
    tt["duration_s"] = tt["t1"] - tt["t0"]
    tt["net_dx"] = tt["x_exit"] - tt["x_entry"]
    tt["parked"] = (tt["v_max"] < 3) & (tt["net_dx"].abs() < 3)
    return tt


def id_switch_estimate(tt, born, died):
    """Share of mid-road track ends followed by a mid-road start that continues the same motion."""
    ends = tt[died]
    starts = tt[born].sort_values("t0")
    st_t, st_dir = starts["t0"].to_numpy(), starts["dir"].to_numpy()
    st_x, st_y = starts["x_entry"].to_numpy(), starts["y_entry"].to_numpy()
    matched = 0
    for t1, d, xe, ye, ve in ends[["t1", "dir", "x_exit", "y_exit", "v_end"]].itertuples(index=False):
        lo, hi = np.searchsorted(st_t, [t1, t1 + SWITCH_DT_S])
        if hi <= lo:
            continue
        sel = slice(lo, hi)
        expected = xe + d * (ve / 3.6) * (st_t[sel] - t1)
        ok = (st_dir[sel] == d) & (np.abs(st_x[sel] - expected) < SWITCH_DX_M) & (np.abs(st_y[sel] - ye) < 1.5)
        matched += bool(ok.any())
    return matched, len(ends)


def best_paper_match(speed_stats, comp_shares):
    """Which paper row do this file's speed statistics and class mix resemble most?"""
    sp = {k: abs(speed_stats["mean"] - r[0]) / r[0] + abs(speed_stats["sd"] - r[2]) / r[2]
          for k, r in PAPER_T2.items()}
    cp = {k: float(np.abs(np.asarray(comp_shares) - np.asarray(r[1])).sum()) for k, r in PAPER_T1.items()}
    return min(sp, key=sp.get), min(cp, key=cp.get)


# %% 5 · Pairwise scan: footprint overlaps, leaders, side neighbours (subsampled frames)
def interaction_scan(df, dt_nom):
    """Same-direction vehicles only; each direction is turned so that travel is +x."""
    step = max(1, int(round(INTERACTION_STEP_S / dt_nom)))
    cols = ["frame", "dir", "cls", "xl", "yl", "L", "W", "v_ms"]
    d = df.loc[(df["frame"] % step == 0) & (df["dir"] != 0), cols].copy()
    if d["v_ms"].isna().all():
        d["v_ms"] = df.loc[d.index, "v_sg"] / 3.6
    d["xl"] = d["xl"] * d["dir"]
    over, lead, side = [], [], []
    n_pairs = 0
    for (fr, _dir), g in d.groupby(["frame", "dir"], sort=False):
        if len(g) < 2:
            continue
        X = g["xl"].to_numpy(); Y = g["yl"].to_numpy()
        ok = np.isfinite(X) & np.isfinite(Y)
        if ok.sum() < 2:
            continue
        g = g[ok]; X, Y = X[ok], Y[ok]
        L = g["L"].to_numpy(); W = g["W"].to_numpy(); V = g["v_ms"].to_numpy(); C = g["cls"].to_numpy()
        pr = cKDTree(np.c_[X, Y]).query_pairs(NEIGHBOUR_RADIUS, output_type="ndarray")
        if len(pr) == 0:
            continue
        i, j = pr[:, 0], pr[:, 1]
        dx, dy = X[j] - X[i], Y[j] - Y[i]
        hl, hw = (L[i] + L[j]) / 2, (W[i] + W[j]) / 2
        lon_ov, lat_ov = np.abs(dx) < hl, np.abs(dy) < hw
        n_pairs += len(i)
        b = lon_ov & lat_ov
        if b.any():
            over.append(pd.DataFrame({"frame": fr, "ci": C[i[b]], "cj": C[j[b]],
                                      "pen_lon": hl[b] - np.abs(dx[b]), "pen_lat": hw[b] - np.abs(dy[b])}))
        m = lat_ov & ~lon_ov
        if m.any():
            ii, jj, ddx = i[m], j[m], dx[m]
            fol = np.where(ddx > 0, ii, jj); lea = np.where(ddx > 0, jj, ii)
            gap = np.abs(ddx) - hl[m]
            o = np.lexsort((gap, fol)); fol, lea, gap = fol[o], lea[o], gap[o]
            _, f1 = np.unique(fol, return_index=True)
            fol, lea, gap = fol[f1], lea[f1], gap[f1]
            dv = V[fol] - V[lea]
            with np.errstate(divide="ignore", invalid="ignore"):
                ttc = np.where(dv > 0.1, gap / dv, np.inf)
            lead.append(pd.DataFrame({"frame": fr, "f_cls": C[fol], "l_cls": C[lea], "gap_m": gap,
                                      "dv_ms": dv, "ttc_s": ttc, "v_f_kmh": V[fol] * 3.6}))
        m2 = lon_ov & ~lat_ov
        if m2.any():
            a = np.r_[i[m2], j[m2]]; bb = np.r_[j[m2], i[m2]]
            c = np.r_[np.abs(dy[m2]) - hw[m2], np.abs(dy[m2]) - hw[m2]]
            o = np.lexsort((c, a)); a, bb, c = a[o], bb[o], c[o]
            _, f1 = np.unique(a, return_index=True)
            side.append(pd.DataFrame({"frame": fr, "s_cls": C[a[f1]], "n_cls": C[bb[f1]],
                                      "clearance_m": c[f1], "v_s_kmh": V[a[f1]] * 3.6}))
    cat = lambda L_: pd.concat(L_, ignore_index=True) if L_ else pd.DataFrame()
    return cat(over), cat(lead), cat(side), n_pairs


# %% 6 · Fundamental diagram (Edie's generalised definitions over the core stretch)
def edie_fd(df, dt_nom):
    """Main travel direction only, so the two carriageways are not added together."""
    lo, hi = np.nanpercentile(df["xl"], 2) + 5, np.nanpercentile(df["xl"], 98) - 5
    Lm = hi - lo
    d = df[(df["xl"] >= lo) & (df["xl"] <= hi) & (df["dir"] == 1)].copy()
    v = d["v_ms"] if d["v_ms"].notna().any() else d["v_sg"] / 3.6
    dt = dt_nom
    d["tdt"] = v * dt
    d["win"] = (d["t"] // FD_WINDOW_S).astype(int)
    agg = d.groupby("win").agg(tdt=("tdt", "sum"), n=("tdt", "size"))
    T = FD_WINDOW_S
    agg["flow_vph"] = agg["tdt"] / (Lm * T) * 3600
    agg["density_vpkm"] = agg["n"] * dt / (Lm * T) * 1000
    agg["speed_kmh"] = agg["flow_vph"] / agg["density_vpkm"]
    agg["stretch_m"] = Lm
    return agg.reset_index()


# %% 7 · Plot helpers
def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.show() if not os.environ.get("EDA_HEADLESS") else None
    plt.close(fig)
    return p


def boxplot(ax, data, labels):
    """Version-safe boxplot (matplotlib renamed `labels` to `tick_labels`)."""
    data = [d if len(d) else np.array([np.nan]) for d in data]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)


def classes_present(df):
    return [c for c in CLASS_ORDER if c in set(df["cls"])] + \
           [c for c in sorted(set(df["cls"])) if c not in CLASS_ORDER]


def plot_site(df, tt, site, fd, lead, side):
    cl = classes_present(df)
    # A · distributions of reported variables
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    for a_, col, lab in zip(axs, ["speed", "lon_acc", "lat_acc"],
                            ["Speed (km/h)", "Longitudinal acc. (m/s²)", "Lateral acc. (m/s²)"]):
        if col in df and df[col].notna().any():
            v = df[col].dropna()
            a_.hist(v, bins=120, color="#3b6ea5")
            a_.set_yscale("log"); a_.set_xlabel(lab); a_.set_ylabel("rows (log)")
    fig.suptitle(f"{site} · reported variables (all rows, log counts expose tails)")
    savefig(fig, f"{site}_01_distributions.png")

    # B · by class
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for a_, col, lab in zip(axs, ["speed", "lon_acc", "lat_acc"], ["Speed (km/h)", "Lon. acc.", "Lat. acc."]):
        if col in df and df[col].notna().any():
            data = [df.loc[df["cls"] == c, col].dropna().to_numpy() for c in cl]
            boxplot(a_, data, cl)
            a_.set_title(lab)
    fig.suptitle(f"{site} · by vehicle class (whiskers 1.5 IQR, outliers hidden)")
    savefig(fig, f"{site}_02_by_class.png")

    # C · lateral placement (lane-keeping)
    bins = np.arange(0, np.nanpercentile(df["yl"], 99.5) + 0.5, 0.5)
    H = np.vstack([np.histogram(df.loc[df["cls"] == c, "yl"].dropna(), bins=bins)[0] for c in cl]).astype(float)
    H = H / np.maximum(H.sum(1, keepdims=True), 1)
    fig, a_ = plt.subplots(figsize=(10, 0.5 * len(cl) + 1.5))
    im = a_.imshow(H, aspect="auto", cmap="Blues", extent=[bins[0], bins[-1], len(cl) - 0.5, -0.5])
    a_.set_yticks(range(len(cl))); a_.set_yticklabels(cl); a_.set_xlabel("lateral position yl (m, 0 = one road edge)")
    plt.colorbar(im, ax=a_, label="share of class rows")
    a_.set_title(f"{site} · lateral placement by class")
    savefig(fig, f"{site}_03_lateral_placement.png")

    # D · time–space diagram excerpt
    t0 = df["t"].min() + min(60, (df["t"].max() - df["t"].min()) / 4)
    sub = df[(df["t"] >= t0) & (df["t"] < t0 + TS_WINDOW_S) & (df["dir"] == 1)]
    vids = sub["vid"].unique()
    if len(vids) > 2500:
        vids = np.random.default_rng(1).choice(vids, 2500, replace=False)
        sub = sub[sub["vid"].isin(vids)]
    cmap = dict(zip(cl, plt.cm.tab10.colors))
    segs, cols = [], []
    for v, g in sub.groupby("vid"):
        segs.append(np.c_[g["t"].to_numpy(), g["xl"].to_numpy()]); cols.append(cmap.get(g["cls"].iat[0], "k"))
    fig, a_ = plt.subplots(figsize=(14, 5))
    a_.add_collection(LineCollection(segs, colors=cols, linewidths=0.4, alpha=0.7))
    a_.autoscale(); a_.set_xlabel("time (s)"); a_.set_ylabel("longitudinal position xl (m)")
    for c in cl:
        a_.plot([], [], color=cmap[c], label=c)
    a_.legend(ncol=len(cl), fontsize=8, loc="upper left")
    a_.set_title(f"{site} · time–space diagram, main direction ({TS_WINDOW_S} s excerpt) — look for broken lines (fragments) and stop-and-go waves")
    savefig(fig, f"{site}_04_time_space.png")

    # E · kinematic consistency
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    s = df.sample(min(len(df), 300_000), random_state=0)
    if "speed" in s and s["speed"].notna().any():
        axs[0].hexbin(s["v_sg"], s["speed"], gridsize=80, bins="log", mincnt=1)
        m_ = np.nanmax([s["speed"].max(), s["v_sg"].max()])
        axs[0].plot([0, m_], [0, m_], "r--", lw=0.8)
        axs[0].set_xlabel("speed from positions, smoothed (km/h)"); axs[0].set_ylabel("reported speed (km/h)")
    if "lon_acc" in s and s["lon_acc"].notna().any():
        axs[1].hexbin(s["a_sg"].clip(-4, 4), s["lon_acc"].clip(-4, 4), gridsize=80, bins="log", mincnt=1)
        axs[1].plot([-4, 4], [-4, 4], "r--", lw=0.8)
        axs[1].set_xlabel("acc. from positions (m/s²)"); axs[1].set_ylabel("reported lon. acc. (m/s²)")
    rs = df.groupby("cls")["res_y"].std().reindex(cl).fillna(0) * 100
    axs[2].bar(cl, rs.values, color="#c0504d"); axs[2].set_ylabel("lateral position jitter SD (cm)")
    axs[2].set_title(f"noise about {SG_WIN_S} s smoothing")
    fig.suptitle(f"{site} · do reported values agree with the positions?")
    savefig(fig, f"{site}_05_kinematic_consistency.png")

    # F · track integrity
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    dts = df["dt"].dropna()
    axs[0].hist(dts.clip(0, 0.5), bins=100, color="#7f7f7f"); axs[0].set_yscale("log")
    axs[0].axvline(dts.median(), color="r", ls="--"); axs[0].set_xlabel("time step within track (s, clipped at 0.5)")
    axs[1].hist(tt["duration_s"], bins=80, color="#4f81bd"); axs[1].set_xlabel("track duration (s)")
    axs[2].hist([tt["x_entry"], tt["x_exit"]], bins=60, label=["entry", "exit"]); axs[2].legend()
    axs[2].set_xlabel("xl at first / last point (m) — mass away from the ends = fragments")
    fig.suptitle(f"{site} · sampling and track integrity")
    savefig(fig, f"{site}_06_track_integrity.png")

    # G · fundamental diagram
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].scatter(fd["density_vpkm"], fd["speed_kmh"], s=12); axs[0].set_xlabel("density (veh/km)"); axs[0].set_ylabel("speed (km/h)")
    axs[1].scatter(fd["flow_vph"], fd["speed_kmh"], s=12); axs[1].set_xlabel("flow (veh/h)"); axs[1].set_ylabel("speed (km/h)")
    axs[2].scatter(fd["density_vpkm"], fd["flow_vph"], s=12); axs[2].set_xlabel("density (veh/km)"); axs[2].set_ylabel("flow (veh/h)")
    fig.suptitle(f"{site} · fundamental diagram (Edie, {FD_WINDOW_S} s windows, main direction)")
    savefig(fig, f"{site}_07_fundamental_diagram.png")

    # H · interactions
    if len(lead) and len(side):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        data = [lead.loc[lead["f_cls"] == c, "gap_m"].to_numpy() for c in cl]
        boxplot(axs[0], data, cl); axs[0].set_ylabel("gap to leader, bumper-to-bumper (m)")
        data = [side.loc[side["s_cls"] == c, "clearance_m"].to_numpy() for c in cl]
        boxplot(axs[1], data, cl); axs[1].set_ylabel("lateral clearance to side neighbour (m)")
        fin = lead["ttc_s"][np.isfinite(lead["ttc_s"])]
        if len(fin):
            xs = np.sort(fin.clip(upper=20)); axs[2].plot(xs, np.arange(1, len(xs) + 1) / len(lead))
            axs[2].set_xlabel("1D TTC to leader (s, closing pairs)"); axs[2].set_ylabel("share of follower-frames")
        fig.suptitle(f"{site} · interaction readiness (same-direction pairs every {INTERACTION_STEP_S} s)")
        savefig(fig, f"{site}_08_interactions.png")


# %% 8 · Per-site analysis
def describe(series):
    s = series.dropna()
    if not len(s):
        return dict(mean=np.nan, median=np.nan, sd=np.nan, min=np.nan, max=np.nan, range=np.nan,
                    p01=np.nan, p99=np.nan)
    return dict(mean=s.mean(), median=s.median(), sd=s.std(), min=s.min(), max=s.max(),
                range=s.max() - s.min(), p01=s.quantile(0.01), p99=s.quantile(0.99))


def analyse_site(path):
    fname = os.path.basename(path)
    site = site_name(fname)
    print(f"\n=== {site}  ({fname}, {os.path.getsize(path) / 1e6:.1f} MB) ===")
    raw, fmt = load_raw(path)
    S = {"site": site, "file": fname, "format": fmt, "raw_rows": len(raw), "raw_columns": list(map(str, raw.columns))}

    # schema table
    sch = pd.DataFrame({"column": raw.columns.astype(str), "dtype": raw.dtypes.astype(str).values,
                        "missing_%": (raw.isna().mean() * 100).round(3).values,
                        "n_unique": [raw[c].nunique(dropna=True) for c in raw.columns],
                        "example": [raw[c].dropna().iloc[0] if raw[c].notna().any() else None for c in raw.columns]})
    sch.to_csv(os.path.join(TAB, f"{site}_schema.csv"), index=False)
    print(sch.to_string(index=False))

    df, meta = standardise(raw, site)
    del raw
    S.update(meta)

    # duplicates and missing core values
    dup = df.duplicated(["vid", "t"]).sum()
    S["duplicate_vid_time_rows"] = int(dup)
    if dup:
        flag(site, "HIGH", "duplicates", f"{dup} rows share vehicle ID and timestamp (dropped for derivatives)")
        df = df.drop_duplicates(["vid", "t"]).reset_index(drop=True)
    core_missing = df[["t", "x", "y"]].isna().any(axis=1).mean()
    S["core_missing_share"] = float(core_missing)
    if core_missing > 0:
        flag(site, "WARN", "missing", f"{core_missing:.3%} rows lack time or position")

    dt_nom = meta["dt_nom"]
    df = derive_kinematics(df, dt_nom)
    tt = track_table(df)
    tt.to_csv(os.path.join(TAB, f"{site}_tracks.csv"))

    # ---- sampling --------------------------------------------------------------------
    dts = df["dt"].dropna()
    gap_s = max(0.2, 2.5 * dt_nom)
    S["dt_median_s"] = float(dts.median())
    S["dt_off_nominal_share"] = float((np.abs(dts - dt_nom) > 0.1 * dt_nom).mean())
    S["gaps_gt_2.5_steps"] = int((dts > gap_s).sum())
    S["tracks_with_gaps"] = int((tt["dt_max"] > gap_s).sum())
    if S["dt_off_nominal_share"] > 0.01:
        flag(site, "WARN", "sampling", f"{S['dt_off_nominal_share']:.2%} of steps deviate >10% from {dt_nom:.4f} s")
    if S["tracks_with_gaps"]:
        flag(site, "WARN", "sampling", f"{S['tracks_with_gaps']} tracks contain gaps > {gap_s:.2f} s")

    # ---- extent vs paper -------------------------------------------------------------
    dur_min = (df["t"].max() - df["t"].min()) / 60
    stretch = np.nanpercentile(df["xl"], 99.5) - np.nanpercentile(df["xl"], 0.5)
    width = np.nanpercentile(df["yl"], 99.5) - np.nanpercentile(df["yl"], 0.5)
    ym = df.loc[(df["dir"] == 1) & df["cls"].isin(CLASS_ORDER), "yl"]
    width_main = np.nanpercentile(ym, 99.5) - np.nanpercentile(ym, 0.5)
    dir_counts = tt["dir"].value_counts().to_dict()
    S.update(duration_min=round(dur_min, 2), stretch_m=round(stretch, 1), width_m=round(width, 2),
             width_main_dir_m=round(width_main, 2), tracks_by_direction={int(k): int(v) for k, v in dir_counts.items()},
             n_vehicles=int(len(tt)), n_rows=int(len(df)),
             nmt_tracks={c: int((tt["cls"] == c).sum()) for c in NMT_CLASSES})
    if site in PAPER_T1:
        pd_, comp, pst, pw = PAPER_T1[site]
        if abs(dur_min - pd_) / pd_ > 0.10:
            flag(site, "WARN", "paper T1", f"duration {dur_min:.1f} min vs paper {pd_} min")
        if abs(stretch - pst) / pst > 0.15:
            flag(site, "INFO", "paper T1", f"observed stretch {stretch:.0f} m vs paper {pst} m (units/axis check)")
        if abs(width_main - pw) / pw > 0.25:
            flag(site, "INFO", "paper T1", f"main-direction lateral extent {width_main:.1f} m vs paper road width {pw} m")
    if dir_counts.get(-1, 0) > 0.05 * len(tt):
        flag(site, "INFO", "directions", f"{dir_counts.get(-1, 0)} tracks travel in the opposite direction "
             f"(other carriageway or U-turns); analysed separately, all-direction lateral extent {width:.1f} m")

    # ---- composition vs paper --------------------------------------------------------
    ttm = tt[tt["cls"].isin(CLASS_ORDER)]          # motorised only, as in the paper
    comp_obs = (ttm["cls"].value_counts(normalize=True) * 100).reindex(CLASS_ORDER).fillna(0)
    comp_long = (ttm.loc[ttm["duration_s"] >= 2, "cls"].value_counts(normalize=True) * 100).reindex(CLASS_ORDER).fillna(0)
    ctab = pd.DataFrame({"observed_%_all_tracks": comp_obs.round(1), "observed_%_tracks_>=2s": comp_long.round(1)})
    if site in PAPER_T1:
        ctab["paper_%"] = PAPER_T1[site][1]
        ctab["diff_pp"] = (ctab["observed_%_all_tracks"] - ctab["paper_%"]).round(1)
        worst = ctab["diff_pp"].abs().max()
        if worst > 5:
            flag(site, "WARN", "paper T1", f"class shares differ from paper by up to {worst:.1f} percentage points")
    ctab.index.name = "class"
    ctab.to_csv(os.path.join(TAB, f"{site}_composition_vs_paper.csv"))
    S["composition"] = ctab.reset_index().to_dict("records")

    # ---- Table 2 replication -----------------------------------------------------------
    sp = describe(df["speed"]) if "speed" in df else describe(df["v_sg"])
    ac = describe(df["lon_acc"]) if "lon_acc" in df else describe(df["a_sg"])
    t2 = pd.DataFrame({"speed_obs": sp, "acc_obs": ac})
    if site in PAPER_T2:
        r = PAPER_T2[site]
        t2["speed_paper"] = pd.Series({"mean": r[0], "median": r[1], "sd": r[2], "range": r[3]})
        t2["acc_paper"] = pd.Series({"mean": r[4], "median": r[5], "sd": r[6], "range": r[7]})
        if abs(sp["mean"] - r[0]) / max(abs(r[0]), 1) > 0.10:
            flag(site, "HIGH", "paper T2", f"mean speed {sp['mean']:.2f} km/h vs paper {r[0]} km/h")
        if ac["range"] > 1.5 * r[7]:
            flag(site, "INFO", "paper T2", f"raw acceleration range {ac['range']:.2f} vs paper {r[7]} "
                 "— paper statistics were probably computed on trimmed/filtered data")
    S["best_paper_row_by_speed"], S["best_paper_row_by_composition"] = best_paper_match(sp, comp_obs.to_numpy())
    if site in PAPER_T2 and S["best_paper_row_by_speed"] != site:
        flag(site, "HIGH", "paper labels", f"speed statistics match paper Table 2 row {S['best_paper_row_by_speed']}, "
             f"not {site}")
    if site in PAPER_T1 and S["best_paper_row_by_composition"] != site:
        flag(site, "WARN", "paper labels", f"class mix matches paper Table 1 row {S['best_paper_row_by_composition']}, "
             f"not {site}")
    t2.index.name = "stat"
    t2.round(4).to_csv(os.path.join(TAB, f"{site}_table2_vs_paper.csv"))
    S["table2"] = t2.round(4).reset_index().to_dict("records")

    # ---- per-class kinematics ----------------------------------------------------------
    def p01(s): return s.quantile(0.01)
    def p99(s): return s.quantile(0.99)
    agg_cols = {c: ["mean", "median", "std", p01, p99]
                for c in ["speed", "lon_acc", "lat_acc", "v_sg", "a_sg", "alat_sg"] if c in df}
    kc = df.groupby("cls").agg(agg_cols)
    kc.columns = [f"{a}_{b}" for a, b in kc.columns]
    kc.round(4).to_csv(os.path.join(TAB, f"{site}_kinematics_by_class.csv"))

    # ---- kinematic consistency -----------------------------------------------------------
    kin = {}
    if "speed" in df:
        m_ = df[["speed", "v_sg"]].dropna()
        m_ = m_[m_["v_sg"] > 1]
        kin["speed_corr"] = float(m_.corr().iloc[0, 1]) if len(m_) > 10 else np.nan
        kin["speed_median_abs_diff_kmh"] = float((m_["speed"] - m_["v_sg"]).abs().median()) if len(m_) else np.nan
        kin["speed_ratio_reported_to_positions"] = float(m_["speed"].median() / m_["v_sg"].median()) if len(m_) else np.nan
        if kin["speed_ratio_reported_to_positions"] == kin["speed_ratio_reported_to_positions"]:
            rr = kin["speed_ratio_reported_to_positions"]
            if not 0.85 < rr < 1.15:
                flag(site, "HIGH", "units", f"reported speed / position-derived speed = {rr:.2f} "
                     "(≈0.28 → m/s, ≈3.6 → positions not in metres)")
        if kin["speed_corr"] < 0.9:
            flag(site, "WARN", "consistency", f"reported speed vs positions correlation {kin['speed_corr']:.3f}")
    if "lon_acc" in df:
        m_ = df[["lon_acc", "a_sg"]].dropna()
        kin["acc_corr"] = float(m_.corr().iloc[0, 1]) if len(m_) > 10 else np.nan
        kin["acc_mae"] = float((m_["lon_acc"] - m_["a_sg"]).abs().mean()) if len(m_) else np.nan
        if kin["acc_corr"] < 0.6:
            flag(site, "WARN", "consistency", f"reported lon. acc. vs positions correlation {kin['acc_corr']:.3f} "
                 "— decide which to trust before modelling")
    if "lat_acc" in df:
        m_ = df[["lat_acc", "alat_sg"]].dropna()
        kin["lat_acc_corr"] = float(m_.corr().iloc[0, 1]) if len(m_) > 10 else np.nan
    kin["pos_noise_sd_x_cm"] = float(df["res_x"].std() * 100)
    kin["pos_noise_sd_y_cm"] = float(df["res_y"].std() * 100)
    kin["raw_vs_smoothed_speed_sd_kmh"] = float((df["v_raw"] - df["v_sg"]).std())
    S["kinematic_consistency"] = kin

    # ---- implausible values ------------------------------------------------------------
    imp = {}
    if "speed" in df:
        imp["speed_gt_100kmh"] = int((df["speed"] > 100).sum()); imp["speed_negative"] = int((df["speed"] < 0).sum())
    if "lon_acc" in df:
        imp["abs_lon_acc_gt_6"] = int((df["lon_acc"].abs() > 6).sum())
    if "lat_acc" in df:
        imp["abs_lat_acc_gt_4"] = int((df["lat_acc"].abs() > 4).sum())
    imp["abs_acc_from_positions_gt_6"] = int((df["a_sg"].abs() > 6).sum())
    imp["jerk_rep_p99_abs"] = float(df["jerk_rep"].abs().quantile(0.99)) if df["jerk_rep"].notna().any() else np.nan
    imp["reversing_rows_share"] = float(((df["vx_sg"] * df["dir"] < -2) & (df["dir"] != 0)).mean())
    if "lon_acc" in df:
        t0m = df["vid"].map(tt["t0"]); t1m = df["vid"].map(tt["t1"])
        near_end = ((df["t"] - t0m) < END_WINDOW_S) | ((t1m - df["t"]) < END_WINDOW_S)
        spike = df["lon_acc"].abs() > 6
        imp["acc_spikes_gt_6"] = int(spike.sum())
        imp["acc_spikes_share_in_first_last_1s"] = float((spike & near_end).sum() / max(spike.sum(), 1))
        imp["rows_share_in_first_last_1s"] = float(near_end.mean())
        if imp["acc_spikes_share_in_first_last_1s"] > 2 * imp["rows_share_in_first_last_1s"]:
            flag(site, "WARN", "acceleration", f"{imp['acc_spikes_share_in_first_last_1s']:.0%} of |acc| > 6 m/s² "
                 f"rows sit in the first/last {END_WINDOW_S} s of a track (those hold "
                 f"{imp['rows_share_in_first_last_1s']:.0%} of rows) → extraction artefacts; trim track ends")
    S["implausible"] = imp
    if imp["reversing_rows_share"] > 0.005:
        flag(site, "WARN", "plausibility", f"{imp['reversing_rows_share']:.2%} rows move backwards >2 km/h")
    if imp.get("speed_gt_100kmh", 0):
        flag(site, "WARN", "plausibility", f"{imp['speed_gt_100kmh']} rows above 100 km/h on an urban mid-block")

    # ---- track integrity ---------------------------------------------------------------
    xa = np.nanpercentile(df["xl"], 1) + EDGE_BUFFER_M
    xb = np.nanpercentile(df["xl"], 99) - EDGE_BUFFER_M
    tmin, tmax = df["t"].min() + 1, df["t"].max() - 1
    mov = tt[(~tt["parked"]) & (tt["dir"] != 0)]
    fwd = mov["dir"] == 1
    born = np.where(fwd, mov["x_entry"] > xa, mov["x_entry"] < xb) & (mov["t0"] > tmin)
    died = np.where(fwd, mov["x_exit"] < xb, mov["x_exit"] > xa) & (mov["t1"] < tmax)
    born = pd.Series(born, index=mov.index); died = pd.Series(died, index=mov.index)
    n_switch, n_ends = id_switch_estimate(mov, born, died)
    integ = dict(
        tracks=int(len(tt)), parked_tracks=int(tt["parked"].sum()),
        short_lt_1s=int((tt["duration_s"] < 1).sum()), short_lt_2s=int((tt["duration_s"] < 2).sum()),
        start_mid_road_share=float(born.mean()), end_mid_road_share=float(died.mean()),
        class_switch_tracks=int((tt["n_classes"] > 1).sum()),
        dimension_unstable_tracks=int(((tt["L_cv"] > 0.1) | (tt["W_cv"] > 0.1)).sum()),
        median_track_duration_s=float(tt["duration_s"].median()),
        mid_road_ends=int(n_ends), ends_continued_by_new_id=int(n_switch),
        id_switch_share_of_mid_road_ends=float(n_switch / max(n_ends, 1)),
    )
    S["track_integrity"] = integ
    if integ["start_mid_road_share"] > 0.10 or integ["end_mid_road_share"] > 0.10:
        flag(site, "HIGH", "fragmentation", f"{integ['start_mid_road_share']:.1%} of moving tracks start and "
             f"{integ['end_mid_road_share']:.1%} end mid-road; {integ['id_switch_share_of_mid_road_ends']:.0%} of those "
             "ends are picked up by a new ID within 1 s (ID switches); the rest are lost tracks or side access")
    if integ["class_switch_tracks"]:
        flag(site, "WARN", "classes", f"{integ['class_switch_tracks']} tracks change class mid-track")
    if integ["dimension_unstable_tracks"] > 0.05 * len(tt):
        flag(site, "WARN", "footprint", f"{integ['dimension_unstable_tracks']} tracks with length/width CV > 10%")

    # footprint plausibility per class
    dims = tt.groupby("cls")[["L_med", "W_med"]].describe(percentiles=[0.05, 0.5, 0.95]).round(2)
    dims.to_csv(os.path.join(TAB, f"{site}_footprints_by_class.csv"))
    if meta.get("length_source") == "file":
        m2 = tt[tt["cls"] == "M2W"]
        if len(m2) and (m2["L_med"] > 3).mean() > 0.05:
            flag(site, "WARN", "footprint", f"{(m2['L_med'] > 3).mean():.1%} of M2W tracks longer than 3 m")

    # ---- interactions --------------------------------------------------------------------
    over, lead, side, n_pairs = interaction_scan(df, dt_nom)
    ov_share = len(over) / max(n_pairs, 1)
    inter = dict(pairs_checked=int(n_pairs), footprint_overlaps=int(len(over)), overlap_share=float(ov_share))
    if len(over):
        ovc = over.groupby(["ci", "cj"]).size().sort_values(ascending=False).rename("overlaps")
        ovc.to_csv(os.path.join(TAB, f"{site}_overlaps_by_class_pair.csv"))
        inter["overlap_median_penetration_lat_m"] = float(over["pen_lat"].median())
    if ov_share > 0.005:
        flag(site, "WARN", "overlap", f"{ov_share:.2%} of nearby vehicle pairs have overlapping footprints "
             "(tracking or dimension error; affects any 2D safety indicator)")
    if len(lead):
        lt = lead.groupby(["f_cls", "l_cls"]).agg(
            n=("gap_m", "size"), gap_median_m=("gap_m", "median"),
            ttc_lt_1_5s_share=("ttc_s", lambda s: (s < 1.5).mean()),
            ttc_lt_3s_share=("ttc_s", lambda s: (s < 3).mean())).round(4)
        lt.to_csv(os.path.join(TAB, f"{site}_leader_follower_by_class_pair.csv"))
        inter["follower_frames"] = int(len(lead))
        inter["ttc_lt_1_5s_share"] = float((lead["ttc_s"] < 1.5).mean())
    if len(side):
        stb = side.groupby(["s_cls", "n_cls"]).agg(
            n=("clearance_m", "size"), clearance_median_m=("clearance_m", "median"),
            clearance_p05_m=("clearance_m", lambda s: s.quantile(0.05))).round(3)
        stb.to_csv(os.path.join(TAB, f"{site}_lateral_clearance_by_class_pair.csv"))
        inter["side_by_side_frames"] = int(len(side))
    S["interactions"] = inter

    # ---- fundamental diagram ---------------------------------------------------------------
    fd = edie_fd(df, dt_nom)
    fd["site"] = site
    fd.to_csv(os.path.join(TAB, f"{site}_fundamental_diagram.csv"), index=False)
    S["fd"] = dict(max_density_vpkm=float(fd["density_vpkm"].max()), max_flow_vph=float(fd["flow_vph"].max()),
                   min_window_speed_kmh=float(fd["speed_kmh"].min()))

    # ---- split planning (leakage control) ---------------------------------------------
    blocks = (tt["t0"] // 600).astype(int)
    S["vehicles_per_10min_block"] = blocks.value_counts().sort_index().to_dict()

    plot_site(df, tt, site, fd, lead, side)

    # ---- standardised copy for the modelling stage (raw values + derived checks) ---------
    keep = [c for c in ["vid", "dir", "frame", "t", "x", "y", "xl", "yl", "speed", "lon_acc", "lat_acc", "cls",
                        "vtype_raw", "L", "W", "v_sg", "a_sg", "alat_sg", "res_x", "res_y"] if c in df]
    try:
        df[keep].to_parquet(os.path.join(STD, f"{site}.parquet"), index=False)
    except Exception as e:
        print("  parquet not written:", e)
    return S, fd


# %% 9 · Run all CSVs
csvs = sorted(glob.glob(os.path.join(BASE, "*.csv")))
print("CSV files found:", [os.path.basename(p) for p in csvs])
found_sites = [site_name(os.path.basename(p)) for p in csvs]
for s in EXPECTED_SITES:
    if s not in found_sites:
        flag(s, "HIGH", "inventory", "no CSV for this site in the folder (paper lists it)")

# smallest files first, so a memory problem on the largest one cannot block the others
csvs = sorted(csvs, key=os.path.getsize)
if ONLY_SITES:
    csvs = [p for p in csvs if site_name(os.path.basename(p)) in ONLY_SITES]
SUMMARIES, FDS = [], []
for p in csvs:
    try:
        S, fd = analyse_site(p)
        SUMMARIES.append(S); FDS.append(fd)
    except MemoryError:
        flag(site_name(os.path.basename(p)), "HIGH", "load", "out of memory — switch to a High-RAM runtime "
             "or set ONLY_SITES to this site and rerun")
    except Exception as e:
        flag(site_name(os.path.basename(p)), "HIGH", "load", f"analysis failed: {e!r}")
    gc.collect()

# %% 10 · Cross-site comparison
rows = []
for S in SUMMARIES:
    k, ti, it = S.get("kinematic_consistency", {}), S.get("track_integrity", {}), S.get("interactions", {})
    t2 = {r["stat"]: r for r in S["table2"]}
    rows.append({
        "site": S["site"], "format": S["format"], "rows": S["n_rows"], "vehicles": S["n_vehicles"],
        "duration_min": S["duration_min"], "paper_duration_min": PAPER_T1.get(S["site"], (np.nan,))[0],
        "sampling_hz": S["sampling_hz"], "stretch_m": S["stretch_m"], "width_main_dir_m": S["width_main_dir_m"],
        "opposite_dir_tracks": S["tracks_by_direction"].get(-1, 0),
        "speed_mean_kmh": round(t2["mean"]["speed_obs"], 2),
        "paper_speed_mean_kmh": PAPER_T2.get(S["site"], (np.nan,))[0],
        "dt_off_nominal_%": round(100 * S["dt_off_nominal_share"], 2),
        "short_tracks_<1s": ti.get("short_lt_1s"), "start_mid_road_%": round(100 * ti.get("start_mid_road_share", np.nan), 1),
        "end_mid_road_%": round(100 * ti.get("end_mid_road_share", np.nan), 1),
        "id_switch_%_of_mid_ends": round(100 * ti.get("id_switch_share_of_mid_road_ends", np.nan), 1),
        "paper_row_by_speed": S["best_paper_row_by_speed"], "paper_row_by_mix": S["best_paper_row_by_composition"],
        "speed_corr_rep_vs_pos": round(k.get("speed_corr", np.nan), 3),
        "acc_corr_rep_vs_pos": round(k.get("acc_corr", np.nan), 3),
        "lat_noise_sd_cm": round(k.get("pos_noise_sd_y_cm", np.nan), 2),
        "overlap_%": round(100 * it.get("overlap_share", np.nan), 3),
        "ttc_<1.5s_%": round(100 * it.get("ttc_lt_1_5s_share", np.nan), 2),
    })
CROSS = pd.DataFrame(rows)
CROSS.to_csv(os.path.join(TAB, "ALL_sites_quality_summary.csv"), index=False)
print(CROSS.to_string(index=False))

if len(CROSS):
    slowest = CROSS.loc[CROSS["speed_mean_kmh"].idxmin(), "site"]
    print(f"Most congested file by mean speed: {slowest}")

if FDS:
    allfd = pd.concat(FDS, ignore_index=True)
    allfd.to_csv(os.path.join(TAB, "ALL_sites_fundamental_diagram.csv"), index=False)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for s_, g in allfd.groupby("site"):
        axs[0].scatter(g["density_vpkm"], g["speed_kmh"], s=10, label=s_)
        axs[1].scatter(g["flow_vph"], g["speed_kmh"], s=10)
        axs[2].scatter(g["density_vpkm"], g["flow_vph"], s=10)
    axs[0].set_xlabel("density (veh/km)"); axs[0].set_ylabel("speed (km/h)"); axs[0].legend()
    axs[1].set_xlabel("flow (veh/h)"); axs[1].set_ylabel("speed (km/h)")
    axs[2].set_xlabel("density (veh/km)"); axs[2].set_ylabel("flow (veh/h)")
    fig.suptitle("All sites · fundamental diagrams (compare with paper Fig. 8)")
    savefig(fig, "ALL_fundamental_diagrams.png")

    # composition comparison
    comp = []
    for S in SUMMARIES:
        for r in S["composition"]:
            comp.append({"site": S["site"], "class": r["class"], "observed": r["observed_%_all_tracks"],
                         "paper": r.get("paper_%", np.nan)})
    comp = pd.DataFrame(comp)
    fig, axs = plt.subplots(1, len(SUMMARIES), figsize=(4 * len(SUMMARIES), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for a_, (s_, g) in zip(axs, comp.groupby("site")):
        xx = np.arange(len(g))
        a_.bar(xx - 0.2, g["observed"], 0.4, label="this file"); a_.bar(xx + 0.2, g["paper"], 0.4, label="paper T1")
        a_.set_xticks(xx); a_.set_xticklabels(g["class"]); a_.set_title(s_)
    axs[0].set_ylabel("% of vehicles"); axs[0].legend()
    fig.suptitle("Class composition: files vs paper Table 1")
    savefig(fig, "ALL_composition_vs_paper.png")

# %% 11 · Videos (metadata + one mid-clip frame each)
VIDEO = []
for vp in sorted(glob.glob(os.path.join(BASE, "*.mp4"))):
    try:
        import cv2
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS); n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH); h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(n // 2), 0)); ok, frame = cap.read(); cap.release()
        name = os.path.splitext(os.path.basename(vp))[0]
        if ok:
            cv2.imwrite(os.path.join(FIG, f"VIDEO_{name}_midframe.jpg"), frame)
        rec = {"video": os.path.basename(vp), "site_guess": site_name(os.path.basename(vp)), "fps": round(fps, 2),
               "frames": int(n), "duration_min": round(n / fps / 60, 2) if fps else np.nan,
               "resolution": f"{int(w)}x{int(h)}"}
        VIDEO.append(rec); print(rec)
        csv_dur = next((S["duration_min"] for S in SUMMARIES if S["site"] == rec["site_guess"]), None)
        if csv_dur and rec["duration_min"] == rec["duration_min"] and rec["duration_min"] < 0.5 * csv_dur:
            flag(rec["site_guess"], "INFO", "video", f"video is {rec['duration_min']} min vs {csv_dur} min of "
                 "trajectories — a sample clip, usable for visual checks only")
    except Exception as e:
        flag(os.path.basename(vp), "WARN", "video", f"could not read: {e!r}")
pd.DataFrame(VIDEO).to_csv(os.path.join(TAB, "videos_metadata.csv"), index=False)

# %% 12 · Report
def md_table(d):
    try:
        return d.to_markdown(index=False)
    except Exception:
        return "```\n" + d.to_string(index=False) + "\n```"


FL = pd.DataFrame(FLAGS)
sev_rank = {"HIGH": 0, "WARN": 1, "INFO": 2}
if len(FL):
    FL = FL.sort_values(by="severity", key=lambda s: s.map(sev_rank)).reset_index(drop=True)
    FL.to_csv(os.path.join(TAB, "ALL_flags.csv"), index=False)

lines = [f"# EDA report — UAV trajectory datasets ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"Input: `{BASE}`  ·  sampling interval detected per file  ·  smoothing: Savitzky–Golay "
         f"{SG_WIN_S} s (acceleration {SG_WIN_ACC_S} s) — diagnostics only; raw values untouched", "",
         "## 1. Quality summary across sites", "", md_table(CROSS) if len(CROSS) else "_no sites analysed_", "",
         "## 2. Flags (fix or justify before modelling)", "", md_table(FL) if len(FL) else "_none_", ""]
for S in SUMMARIES:
    s = S["site"]
    lines += [f"## Site {s}", "",
              f"- File `{S['file']}` · {S['format']} · {S['n_rows']:,} rows · {S['n_vehicles']:,} vehicles",
              f"- Column mapping: `{S['mapping']}` · time unit: {S['time_unit']} · sampling {S['sampling_hz']} Hz",
              f"- Tracks by direction (1 = main, -1 = opposite, 0 = static): `{S['tracks_by_direction']}` · "
              f"non-motorised tracks: `{S['nmt_tracks']}` · main-direction width {S['width_main_dir_m']} m",
              f"- Closest paper rows: Table 2 speed → {S['best_paper_row_by_speed']}, "
              f"Table 1 class mix → {S['best_paper_row_by_composition']}",
              f"- Duration {S['duration_min']} min · stretch {S['stretch_m']} m · lateral extent {S['width_m']} m · "
              f"road axis {S['axis_angle_deg']}° · wrong-way share {S['wrong_way_share']:.2%}",
              f"- Track integrity: `{S['track_integrity']}`",
              f"- Kinematic consistency: `{ {k: round(v, 4) for k, v in S['kinematic_consistency'].items()} }`",
              f"- Implausible values: `{S['implausible']}`",
              f"- Interactions: `{S['interactions']}`",
              f"- Fundamental diagram extremes: `{S['fd']}`",
              f"- Vehicles starting in each 10-min block (for time-block splits): `{S['vehicles_per_10min_block']}`",
              "", "Composition vs paper Table 1:", "", md_table(pd.DataFrame(S["composition"])), "",
              "Table 2 replication:", "", md_table(pd.DataFrame(S["table2"])), "",
              *[f"![{f}](figures/{f})" for f in sorted(os.listdir(FIG)) if f.startswith(s + "_")], ""]
if VIDEO:
    lines += ["## Videos", "", md_table(pd.DataFrame(VIDEO)), ""]
lines += ["## How to read this before modelling", "",
          "- **HIGH** flags block modelling until resolved (missing site, unit errors, heavy fragmentation, "
          "duplicates, mismatch with the paper's speeds).",
          "- Use the position-derived vs reported comparisons to decide which kinematic channel to model.",
          "- Footprint overlaps and lateral jitter set the floor for any 2D gap, clearance or TTC analysis.",
          "- Split train/test by site or by the 10-minute blocks listed above, never by random rows."]
with open(os.path.join(OUT, "EDA_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


with open(os.path.join(OUT, "eda_flags.json"), "w") as f:
    json.dump(_clean({"flags": FLAGS, "sites": SUMMARIES, "videos": VIDEO}), f, indent=2, default=str)

print("\nDone. Report:", os.path.join(OUT, "EDA_report.md"))
print("Flags:", len(FLAGS), "| figures:", len(os.listdir(FIG)), "| tables:", len(os.listdir(TAB)))
