# %% [markdown]
# # Paper exhibits — audit of open UAV trajectories and two-wheeler filtering (TRR manuscript)
#
# Builds every figure and table of the manuscript and its supplement from the cleaned data:
# main-text exhibits (Figures 1–7, Tables 1–5) and supplementary exhibits (Figures S1–S13, Tables S1–S18).
# Part 1 audits all six released files; Part 2 analyzes two-wheeler filtering on the main carriageways of
# DEL_2 (congested) and DEL_3 (uncongested reference); the supplement adds audit details, filtering details,
# and calibration-ready targets for simulation.
#
# Inputs (written by 01_eda and 02_clean): `EDA_results/standardised/*.parquet`, `EDA_results/eda_flags.json`,
# `cleaned/*_clean.parquet`, `cleaned/tables/*.csv`.
# Outputs in `<BASE>/paper_packs/TRR_combined/`: `TRR_combined.pdf` (everything), `TRR_combined_main.pdf`,
# `TRR_combined_supplement.pdf`, `figures/` (PNG and TIFF at 300 dpi, vector PDF), `tables/` (CSV, LaTeX),
# and `index.csv`.

# %% 0 · Setup
import os, re, glob, gc, json, textwrap, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("EDA_HEADLESS"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

BASE = os.environ.get("EDA_BASE", "/content/drive/MyDrive/Research/SAGE_transport")
EDA = os.path.join(BASE, "EDA_results")
STD = os.path.join(EDA, "standardised")
CLEAN = os.path.join(BASE, "cleaned")
OUT = os.path.join(BASE, "paper_packs")
os.makedirs(OUT, exist_ok=True)

# %% 1 · Configuration
RUN_PARTS = ["audit", "filtering", "targets"]   # parts of the combined paper (targets go to the supplement)
PAPER_KEY = "TRR_combined"
PAPER_TITLE = ("Auditing open drone trajectories of area-based traffic: data pitfalls and what cleaned data "
               "reveal about two-wheeler filtering")
SITES = ["DEL_2", "DEL_3", "DEL_4", "DEL_5", "NDA_1", "NDA_2"]
DELHI = ["DEL_2", "DEL_3", "DEL_4", "DEL_5"]
MAIN_CORRIDOR = {s: (1, 0) for s in SITES}   # (direction, corridor) of the main carriageway, from the cleaning report
P4_SITES = {"congested": "DEL_2", "comparison": "DEL_3"}
P2_SITES = ["DEL_2", "DEL_3", "DEL_4", "DEL_5"]
WINDOW_S = 30                     # Edie window
REGIMES = [(0, 15, "congested <15"), (15, 25, "slow 15–25"), (25, 35, "moderate 25–35"), (35, 200, "free >35")]
HARD_BRAKE = -3.0                 # m/s², for the hard-braking comparison
MIN_COVERAGE = 0.95               # Edie windows need ≥95% of their time stamps present (drops windows over data gaps)
# Passing (filtering) definition
PASS_STEP_S = 0.5                 # sampling step for order inversions
PERSIST_S = 2.0                   # a pass counts only if the new order still holds this long afterwards
NEAR_M = 1.5                      # a "close pass" has lateral clearance below this (typical widths)
# Block bootstrap for confidence intervals (consecutive 30 s windows are not independent)
BOOT_BLOCK_WIN = 10               # block = 10 consecutive windows = 5 min
N_BOOT = 1000
# Which edge of the main carriageway is the MEDIAN side, in the road-aligned lateral axis ("low" = small yl,
# "high" = large yl). CHECK each site against the video before using lateral results; None = not confirmed.
MEDIAN_EDGE = {"DEL_2": None, "DEL_3": None, "DEL_4": None, "DEL_5": None}
# Calibration-target rules (supplement)
STOPPED_KMH = 5.0                 # leader-free samples slower than this are stopped/dwelling, not desired speed
FREE_REGIME_MIN_KMH = 25.0        # desired speeds only from windows at or above this space-mean speed
JAM_MAX_GAP_M = 10.0              # standstill gap: follower and leader both stopped, gap below this
SPIKE = 6.0                       # m/s², acceleration spike threshold used in cleaning
TTC_CRIT = 1.5                    # s
LEADER_RADIUS_M = 40.0
PASS_MAX_DS_M = 30.0
WIDTH_SENS_M = 0.10               # ± per vehicle in the footprint sensitivity test
STATED_FPS = 30

CLASSES = ["Car", "M2W", "M3W", "LCV", "Bus", "HCV"]
# Typical vehicle sizes (length, width in m). These are PLACEHOLDERS: replace them with values from a citable
# source (e.g. Indo-HCM 2017 or the PCU literature you cite) and name that source below.
TYPICAL_DIMS_SOURCE = "Bhattacharyya, Maitra, and Boltze 2020"
TYPICAL_DIMS = {"Car": (4.0, 1.7), "M2W": (1.9, 0.7), "M3W": (2.6, 1.3), "LCV": (5.0, 1.9),
                "Bus": (10.5, 2.5), "HCV": (8.5, 2.5), "Bicycle": (1.7, 0.6), "Pedestrian": (0.5, 0.5)}
TYPICAL_LABEL = "typical (placeholder)" if "REPLACE" in TYPICAL_DIMS_SOURCE else "typical (cited)"
# Validated categorical palette (fixed order) + markers so every series also reads in greyscale
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLS_COLOR = dict(zip(CLASSES, PAL))
CLS_MARK = dict(zip(CLASSES, ["o", "s", "^", "D", "v", "P"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.pad_inches": 0.15, "figure.dpi": 110, "legend.frameon": False,
    "axes.titlepad": 10, "axes.labelpad": 6, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.10,
    "figure.constrained_layout.w_pad": 0.10, "figure.constrained_layout.hspace": 0.12,
    "figure.constrained_layout.wspace": 0.10,
})
W1, W2 = 6.5, 3.2   # full and half text width (inches)

# %% 2 · Pack writer (PDF + per-item files + index)
class Pack:
    """Writes every item to the full pack PDF; with split=True, main-text items (Figure 1, Table 1, …) also go to
    <key>_main.pdf and supplementary items are numbered Figure S1, Table S1, …"""

    def __init__(self, key, title, note, split=False, supp_order=None):
        self.key, self.dir, self.split = key, os.path.join(OUT, key), split
        self.supp_order = supp_order or {"F": [], "T": []}
        self.fdir, self.tdir = os.path.join(self.dir, "figures"), os.path.join(self.dir, "tables")
        for d in (self.dir, self.fdir, self.tdir):
            os.makedirs(d, exist_ok=True)
        self.pdf = PdfPages(os.path.join(self.dir, f"{key}.pdf"))
        self.pdf_main = PdfPages(os.path.join(self.dir, f"{key}_main.pdf")) if split else None
        self.items, self.held = [], []
        self.n = {("F", True): 0, ("F", False): 0, ("T", True): 0, ("T", False): 0}
        fig = plt.figure(figsize=(8.27, 11.69), layout=None); fig.patch.set_facecolor("white")
        fig.text(0.08, 0.86, title, fontsize=15, weight="bold", wrap=True, va="top")
        fig.text(0.08, 0.74, textwrap.fill(note, 95), fontsize=9, va="top", color=INK2)
        fig.text(0.08, 0.08, f"Generated {datetime.now():%Y-%m-%d %H:%M} from {BASE}", fontsize=7, color=INK2)
        self._save(fig, True); plt.close(fig)

    def _save(self, fig, main, code=None):
        if self.split and not main and code is not None:
            self.held.append((code, len(self.held), fig))      # supplementary pages are written sorted at close()
            return
        self.pdf.savefig(fig)
        if self.pdf_main is not None and main:
            self.pdf_main.savefig(fig)

    def ref(self, kind, name):
        """Label of a supplementary item fixed in supp_order, e.g. ref('F', 'footprints') -> 'Figure S4'."""
        word = "Figure" if kind == "F" else "Table"
        return f"{word} S{self.supp_order[kind].index(name) + 1}"

    def _label(self, kind, main, name=None):
        main = main or not self.split
        word = "Figure" if kind == "F" else "Table"
        if not main and name in self.supp_order.get(kind, []):
            k = self.supp_order[kind].index(name) + 1
            return f"{word} S{k}", f"{kind}S{k:02d}"
        self.n[(kind, main)] += 1
        k = self.n[(kind, main)] + (0 if main else len(self.supp_order.get(kind, [])))
        return (f"{word} {k}", f"{kind}{k:02d}") if main else (f"{word} S{k}", f"{kind}S{k:02d}")

    def figure(self, fig, name, caption, main=True):
        label, code = self._label("F", main, name)
        stem = os.path.join(self.fdir, f"{code}_{name}")
        fig.patch.set_facecolor("white")
        for a in fig.axes:
            a.set_facecolor("white")
        fig.savefig(stem + ".png", bbox_inches="tight", facecolor="white")
        fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(stem + ".tif", bbox_inches="tight", facecolor="white", dpi=300)   # TRR upload: one file per figure
        if not os.environ.get("EDA_HEADLESS"):
            plt.show()
        plt.close(fig)
        from PIL import Image
        img = np.asarray(Image.open(stem + ".png").convert("RGB")).copy()
        img[(img >= 248).all(axis=2)] = 255          # near-white → pure white (no grey panel behind figures)
        ih, iw = img.shape[:2]
        page = plt.figure(figsize=(8.27, 11.69), layout=None); page.patch.set_facecolor("white")
        max_w, max_h, top = 0.84, 0.62, 0.93
        w = max_w; h = w * (ih / iw) * (8.27 / 11.69)
        if h > max_h:
            h = max_h; w = h * (iw / ih) * (11.69 / 8.27)
        ax = page.add_axes([(1 - w) / 2, top - h, w, h]); ax.imshow(img, interpolation="none"); ax.axis("off")
        page.text((1 - max_w) / 2, top - h - 0.035, textwrap.fill(f"{label}. {caption}", 105),
                  fontsize=9, va="top", ha="left", color=INK, linespacing=1.4)
        self._save(page, main or not self.split, code); plt.close(page)
        self.items.append({"item": label, "placement": "main" if (main or not self.split) else "supplementary",
                           "name": name, "caption": caption, "file": stem + ".png"})

    def table(self, df, name, caption, rows_per_page=22, max_cols=11, digits=2, main=True):
        label, code = self._label("T", main, name)
        stem = os.path.join(self.tdir, f"{code}_{name}")
        df = df.copy()
        df.to_csv(stem + ".csv", index=False)
        try:
            df.to_latex(stem + ".tex", index=False, float_format=lambda v: f"{v:.{digits}f}",
                        caption=f"{caption}", label=f"tab:{name}")
        except Exception as e:
            print("  (LaTeX export skipped:", e, ")")
        show = df.copy()
        wrap = max(12, min(40, int(150 / max(1, min(len(show.columns), max_cols)))))
        for c in show.columns:
            if pd.api.types.is_float_dtype(show[c]):
                show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
            else:
                show[c] = show[c].map(lambda v: "" if (v is None or (isinstance(v, float) and np.isnan(v)))
                                      else textwrap.fill(str(v), wrap))
        cols = list(show.columns)
        col_chunks = [cols] if len(cols) <= max_cols else \
            [[cols[0]] + cols[i:i + max_cols - 1] for i in range(1, len(cols), max_cols - 1)]
        pages = [(r, cc) for cc in col_chunks for r in range(0, max(len(show), 1), rows_per_page)]
        for k, (r, cc) in enumerate(pages):
            part = show.iloc[r:r + rows_per_page][cc]
            fig, ax = plt.subplots(figsize=(11.69, 8.27), layout=None); ax.axis("off")   # landscape A4
            cont = f" (part {k + 1} of {len(pages)})" if len(pages) > 1 else ""
            ax.set_title(textwrap.fill(f"{label}{cont}. {caption}", 140), loc="left", fontsize=9, pad=12)
            if len(part):
                heads = [textwrap.fill(str(c), 16) for c in part.columns]
                nh = max(h.count("\n") + 1 for h in heads)
                tb = ax.table(cellText=part.values, colLabels=heads,
                              loc="upper left", cellLoc="left", colLoc="left")
                tb.auto_set_font_size(False); tb.set_fontsize(7.5)
                nl = part.apply(lambda col: col.map(lambda v: str(v).count("\n") + 1)).max(axis=1).to_numpy()
                for (i, j), cell in tb.get_celld().items():
                    cell.set_height(0.026 * (nh + 0.6) if i == 0 else 0.026 * (int(nl[i - 1]) + 0.6))
                    cell.set_edgecolor(GRID)
                    if i == 0:
                        cell.set_text_props(weight="bold"); cell.set_facecolor("#f2f1ee")
                tb.auto_set_column_width(list(range(len(part.columns))))
            self._save(fig, main or not self.split, code); plt.close(fig)
        self.items.append({"item": label, "placement": "main" if (main or not self.split) else "supplementary",
                           "name": name, "caption": caption, "file": stem + ".csv"})

    @staticmethod
    def _key(item):
        m = re.match(r"(Figure|Table) S?(\d+)", item)
        return (0 if m.group(1) == "Figure" else 1, int(m.group(2)))

    def _contents(self, title, sub):
        pages, fig, y = [], None, 0
        for _, r in sub.iterrows():
            txt = textwrap.fill(f"{r['item']} — {r['caption']}", 110)
            h = 0.022 * (txt.count("\n") + 1) + 0.008
            if fig is None or y - h < 0.03:
                fig, ax = plt.subplots(figsize=(8.27, 11.69), layout=None); ax.axis("off")
                ax.set_title(title + (" (continued)" if pages else ""), loc="left", fontsize=11)
                pages.append(fig); y = 0.97
            ax.text(0, y, txt, fontsize=7.5, va="top", transform=ax.transAxes)
            y -= h
        return pages

    def close(self):
        idx = pd.DataFrame(self.items)
        idx["_k"] = idx["item"].map(self._key)
        idx = pd.concat([idx[idx["placement"] == "main"], idx[idx["placement"] != "main"].sort_values("_k", kind="stable")])
        idx.drop(columns="_k").to_csv(os.path.join(self.dir, "index.csv"), index=False)
        for fig in self._contents("Main-text exhibits" if self.split else "Contents of this pack",
                                  idx[idx["placement"] == "main"]):
            self._save(fig, True); plt.close(fig)
        if self.split:
            pdf_s = PdfPages(os.path.join(self.dir, f"{self.key}_supplement.pdf"))
            sub = idx[idx["placement"] != "main"]
            pages = self._contents("Supplementary exhibits", sub)
            order = sorted(self.held, key=lambda h: (0 if h[0][0] == "F" else 1, int(h[0][2:]), h[1]))
            for fig in pages + [h[2] for h in order]:
                self.pdf.savefig(fig); pdf_s.savefig(fig); plt.close(fig)
            pdf_s.close()
        self.pdf.close()
        if self.pdf_main is not None:
            self.pdf_main.close()
        nm = (idx["placement"] == "main").sum()
        print(f"[{self.key}] {len(idx)} exhibits ({nm} main) → {self.dir}")


# %% 3 · Loaders and shared computations
META = {s["site"]: s for s in json.load(open(os.path.join(EDA, "eda_flags.json")))["sites"]}
SITES = [s for s in SITES if s in META and os.path.exists(os.path.join(CLEAN, f"{s}_clean.parquet"))]
DELHI = [s for s in DELHI if s in SITES]
print("Sites available:", SITES)


def ctab(name):
    fs = sorted(glob.glob(os.path.join(CLEAN, "tables", f"*_{name}.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


CLEAN_SUM = pd.read_csv(os.path.join(CLEAN, "tables", "ALL_cleaning_summary.csv")).set_index("site")
CLEAN_LOG, CORR, MID, FOOT = ctab("cleaning_log"), ctab("corridors"), ctab("mid_road"), ctab("footprints")


def dt_of(site):
    return float(META[site]["dt_nom"])


STRAIGHT = {}      # site -> {"drift_m_per_100m", "width_local_m"} from straightening the main carriageway


def straighten(df, site, bin_m=10.0):
    """Remove a linear drift of the carriageway in the lateral coordinate (road not exactly parallel to the
    principal axis). Fits the centre line of the 1st/99th lateral percentiles in 10 m bins along s."""
    s_, y = df["s"].to_numpy(float), df["yl"].to_numpy(float)
    edges = np.arange(np.nanmin(s_), np.nanmax(s_) + bin_m, bin_m)
    k = np.digitize(s_, edges)
    cs, lo, hi = [], [], []
    for b in np.unique(k):
        m = k == b
        if m.sum() >= 500:
            cs.append(np.median(s_[m])); lo.append(np.percentile(y[m], 1)); hi.append(np.percentile(y[m], 99))
    if len(cs) < 5:
        return df
    cs, lo, hi = map(np.asarray, (cs, lo, hi))
    slope = np.polyfit(cs, (lo + hi) / 2, 1)[0]
    w = hi - lo                                # band width within each 10 m bin (hardly affected by the drift)
    STRAIGHT[site] = {"drift_m_per_100m": 100 * slope, "width_local_m": float(np.median(w))}
    df = df.copy()
    df["yl"] = y - slope * (s_ - np.median(cs))
    return df


def load_clean(site, cols=None, main_only=False):
    df = pd.read_parquet(os.path.join(CLEAN, f"{site}_clean.parquet"), columns=cols)
    if main_only:
        d, c = MAIN_CORRIDOR[site]
        df = df[(df["dir"] == d) & (df["corridor"] == c) & ~df["is_nmt"]]
        if "s" in df and "yl" in df:
            df = straighten(df, site)
    return df


def load_std(site, cols):
    return pd.read_parquet(os.path.join(STD, f"{site}.parquet"), columns=cols)


def typical(cls_series, k):
    return cls_series.map(lambda c: TYPICAL_DIMS.get(c, (4.0, 1.7))[k]).to_numpy(float)


def regime_of(speed_kmh):
    lab = np.full(len(speed_kmh), REGIMES[-1][2], dtype=object)
    for lo, hi, name in REGIMES:
        lab[(speed_kmh >= lo) & (speed_kmh < hi)] = name
    return lab


def edie(d, dt, pos="s", window=WINDOW_S, by_class=False, core=None, dt_true=None):
    """Edie's definitions over the core stretch; returns per-window table (and stretch bounds)."""
    if core is None:
        lo, hi = np.nanpercentile(d[pos], [2, 98]); lo, hi = lo + 5, hi - 5
    else:
        lo, hi = core
    Lm = hi - lo
    x = d[(d[pos] >= lo) & (d[pos] <= hi)]
    g = pd.DataFrame({"win": (x["t"] // window).astype(int).to_numpy(), "cls": x["cls"].to_numpy(),
                      "tdt": (x["speed"] / 3.6 * dt).to_numpy(), "tts": dt})
    tot = g.groupby("win")[["tdt", "tts"]].sum()
    out = pd.DataFrame({"flow_vph": tot["tdt"] / (Lm * window) * 3600,
                        "density_vpkm": tot["tts"] / (Lm * window) * 1000})
    out["speed_kmh"] = out["flow_vph"] / out["density_vpkm"]
    # share of the window's time stamps that are present (independent of the dt used for weighting)
    dtt = dt_true or dt                            # true sampling interval (dt may be a wrongly assumed one)
    fr = np.unique(np.round(d["t"].to_numpy() / dtt).astype(np.int64))
    nt = pd.Series(fr).groupby((fr * dtt // window).astype(int)).size()
    out["coverage"] = (nt.reindex(out.index).fillna(0) / (window / dtt)).clip(upper=1.0)
    if len(out) > 2:
        out = out.iloc[1:-1]                      # drop partial first/last windows
    out = out[out["coverage"] >= MIN_COVERAGE]    # drop windows that straddle a gap in the data
    if by_class:
        pc = g.groupby(["win", "cls"])[["tdt", "tts"]].sum().unstack("cls").fillna(0)
        for c in CLASSES:
            if ("tdt", c) in pc.columns:
                out[f"flow_{c}"] = pc[("tdt", c)] / (Lm * window) * 3600
                out[f"dens_{c}"] = pc[("tts", c)] / (Lm * window) * 1000
                out[f"km_{c}"] = pc[("tdt", c)] / 1000
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[f"speed_{c}"] = out[f"flow_{c}"] / out[f"dens_{c}"]
    out["regime"] = regime_of(out["speed_kmh"].to_numpy())
    return out.reset_index(), (lo, hi)


def frame_slices(d):
    d = d.sort_values("frame", kind="mergesort")
    f = d["frame"].to_numpy()
    u, i0 = np.unique(f, return_index=True)
    return d, u, np.r_[i0, len(f)]


def leader_pairs(d, dt, step_s=0.5, dims="typical"):
    """Nearest same-corridor leader (lateral overlap) and nearest side neighbour per vehicle, every step_s."""
    step = max(1, int(round(step_s / dt)))
    d = d[d["frame"] % step == 0]
    d, u, b = frame_slices(d)
    S = d["s"].to_numpy(); Y = d["yl"].to_numpy(); V = d["speed"].to_numpy() / 3.6
    C = d["cls"].to_numpy(); ID = d["vid"].to_numpy(); T = d["t"].to_numpy()
    L = typical(d["cls"], 0) if dims == "typical" else d["L"].to_numpy()
    W = typical(d["cls"], 1) if dims == "typical" else d["W"].to_numpy()
    lead, side = [], []
    for k in range(len(u)):
        s0, s1 = b[k], b[k + 1]
        if s1 - s0 < 2:
            continue
        pr = cKDTree(np.c_[S[s0:s1], Y[s0:s1]]).query_pairs(LEADER_RADIUS_M, output_type="ndarray")
        if not len(pr):
            continue
        i, j = pr[:, 0] + s0, pr[:, 1] + s0
        dx, dy = S[j] - S[i], Y[j] - Y[i]
        hl, hw = (L[i] + L[j]) / 2, (W[i] + W[j]) / 2
        lat, lon = np.abs(dy) < hw, np.abs(dx) < hl
        m = lat
        if m.any():
            fo = np.where(dx[m] > 0, i[m], j[m]); le = np.where(dx[m] > 0, j[m], i[m])
            gap = np.abs(dx[m]) - hl[m]
            o = np.lexsort((gap, fo)); fo, le, gap = fo[o], le[o], gap[o]
            _, f1 = np.unique(fo, return_index=True)
            fo, le, gap = fo[f1], le[f1], gap[f1]
            lead.append((np.c_[T[fo], ID[fo], ID[le], gap, V[fo], V[le]], C[fo], C[le]))
        m2 = lon & ~lat
        if m2.any():
            a = np.r_[i[m2], j[m2]]; bb = np.r_[j[m2], i[m2]]
            off = np.abs(dy[m2]); off = np.r_[off, off]
            hw2 = np.r_[hw[m2], hw[m2]]
            o = np.lexsort((off - hw2, a)); a, bb, off, hw2 = a[o], bb[o], off[o], hw2[o]
            _, f1 = np.unique(a, return_index=True)
            side.append((np.c_[T[a[f1]], ID[a[f1]], off[f1], hw2[f1], V[a[f1]]], C[a[f1]], C[bb[f1]]))
    if lead:
        arr = np.vstack([x[0] for x in lead])
        LD = pd.DataFrame(arr, columns=["t", "f_vid", "l_vid", "gap_m", "v_f", "v_l"])
        LD["f_cls"] = np.concatenate([x[1] for x in lead]); LD["l_cls"] = np.concatenate([x[2] for x in lead])
        dv = LD["v_f"] - LD["v_l"]
        LD["ttc_s"] = np.where((dv > 0.1) & (LD["gap_m"] > 0), LD["gap_m"] / dv, np.inf)
        LD["time_gap_s"] = np.where(LD["v_f"] > 0.5, LD["gap_m"].clip(lower=0) / LD["v_f"], np.nan)
    else:
        LD = pd.DataFrame(columns=["t", "f_vid", "l_vid", "gap_m", "v_f", "v_l", "f_cls", "l_cls", "ttc_s", "time_gap_s"])
    if side:
        arr = np.vstack([x[0] for x in side])
        SD = pd.DataFrame(arr, columns=["t", "vid", "offset_m", "half_widths_m", "v"])
        SD["cls"] = np.concatenate([x[1] for x in side]); SD["n_cls"] = np.concatenate([x[2] for x in side])
        SD["clearance_m"] = SD["offset_m"] - SD["half_widths_m"]
    else:
        SD = pd.DataFrame(columns=["t", "vid", "offset_m", "half_widths_m", "v", "cls", "n_cls", "clearance_m"])
    return LD, SD


def overtakes(d, dt, core, step_s=PASS_STEP_S, widths="typical"):
    """Order inversions along s between consecutive samples: i was behind j and is now ahead."""
    step = max(1, int(round(step_s / dt)))
    d = d[d["frame"] % step == 0]
    d, u, b = frame_slices(d)
    pos = {f: k for k, f in enumerate(u)}
    S = d["s"].to_numpy(); Y = d["yl"].to_numpy(); V = d["speed"].to_numpy(); C = d["cls"].to_numpy()
    ID = d["vid"].to_numpy(); T = d["t"].to_numpy()
    Wt = d["W"].to_numpy(float) if (widths == "file" and "W" in d) else typical(d["cls"], 1)
    recs = []
    for k, f in enumerate(u):
        k2 = pos.get(f + step)
        if k2 is None:
            continue
        a0, a1 = b[k], b[k + 1]; c0, c1 = b[k2], b[k2 + 1]
        id0, id1 = ID[a0:a1], ID[c0:c1]
        common, i0, i1 = np.intersect1d(id0, id1, return_indices=True)
        if len(common) < 2:
            continue
        g0, g1 = a0 + i0, c0 + i1
        s0, s1 = S[g0], S[g1]
        behind = (s0[:, None] < s0[None, :]) & (np.abs(s0[:, None] - s0[None, :]) < PASS_MAX_DS_M)
        ahead = s1[:, None] > s1[None, :]
        pi, pj = np.nonzero(behind & ahead)
        if not len(pi):
            continue
        keep = (s0[pi] >= core[0]) & (s0[pi] <= core[1])
        pi, pj = pi[keep], pj[keep]
        if not len(pi):
            continue
        off = (np.abs(Y[g0[pi]] - Y[g0[pj]]) + np.abs(Y[g1[pi]] - Y[g1[pj]])) / 2
        hw = (Wt[g0[pi]] + Wt[g0[pj]]) / 2
        recs.append(pd.DataFrame({"t": T[g0[pi]], "t1": T[g1[pi]], "i_vid": ID[g0[pi]], "j_vid": ID[g0[pj]],
                                  "passer_cls": C[g0[pi]], "passed_cls": C[g0[pj]],
                                  "offset_m": off, "half_widths_m": hw, "clearance_m": off - hw,
                                  "v_passer": (V[g0[pi]] + V[g1[pi]]) / 2, "v_passed": (V[g0[pj]] + V[g1[pj]]) / 2,
                                  "y_passer": Y[g0[pi]]}))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def data_gaps(t, vid, dt0):
    """Gaps in the time stamps of a file at sampling interval dt0: (list of (start, length) of file-wide intervals
    with no record, share of within-track steps longer than 1.5 intervals, dt0)."""
    fr = np.unique(np.round(np.asarray(t, float) / dt0).astype(np.int64))
    k = np.nonzero(np.diff(fr) > 1)[0]
    gaps = [(float(fr[i] * dt0), float((fr[i + 1] - fr[i]) * dt0)) for i in k]
    o = np.lexsort((t, vid)); tv, vv = np.asarray(t)[o], np.asarray(vid)[o]
    same = vv[1:] == vv[:-1]
    within = float(((np.diff(tv) > 1.5 * dt0) & same).sum() / max(same.sum(), 1))
    return gaps, within, dt0


def track_resets(t, vid, min_n=20, factor=10.0):
    """Instants where many tracks end AND many start within the same second (ID re-initialisation, e.g. at a
    processing-segment boundary). Returns list of (time_s, n_ends, n_starts)."""
    g = pd.DataFrame({"t": t, "vid": vid}).groupby("vid")["t"].agg(["min", "max"])
    t0, t1 = float(np.min(t)), float(np.max(t))
    sb = np.floor(g["min"]).astype(int).value_counts()
    eb = np.floor(g["max"]).astype(int).value_counts()
    sb = sb[(sb.index > t0 + 1) & (sb.index < t1 - 1)]; eb = eb[(eb.index > t0 + 1) & (eb.index < t1 - 1)]
    base_s, base_e = max(sb.median() if len(sb) else 1, 1), max(eb.median() if len(eb) else 1, 1)
    out = []
    for sec in sorted(set(sb.index) | set(eb.index)):
        ns = int(sb.get(sec, 0) + sb.get(sec + 1, 0)); ne = int(eb.get(sec, 0) + eb.get(sec - 1, 0))
        if ns >= max(min_n, factor * base_s) and ne >= max(min_n, factor * base_e):
            if not out or sec - out[-1][0] > 2:
                out.append((float(sec), ne, ns))
    return out


def track_index(d):
    """vid -> (t, s) arrays, for interpolating positions at arbitrary times."""
    d = d.sort_values(["vid", "t"], kind="mergesort")
    v = d["vid"].to_numpy(); t = d["t"].to_numpy(); s_ = d["s"].to_numpy()
    cut = np.r_[0, np.nonzero(v[1:] != v[:-1])[0] + 1, len(v)]
    return {v[a]: (t[a:b], s_[a:b]) for a, b in zip(cut[:-1], cut[1:])}


def classify_passes(ov, TI, persist_s=PERSIST_S):
    """Adds 'persistent' (new order still holds persist_s later, both tracks present) and 'net' (pair's order at the
    end of their common presence is the reverse of the order at its start; first event of each pair only)."""
    if not len(ov):
        return ov
    pers = np.zeros(len(ov), bool)
    tq = ov["t1"].to_numpy() + persist_s
    for k, (iv, jv, tt) in enumerate(zip(ov["i_vid"].to_numpy(), ov["j_vid"].to_numpy(), tq)):
        ti, si = TI[iv]; tj, sj = TI[jv]
        if tt <= ti[-1] and tt <= tj[-1]:
            pers[k] = np.interp(tt, ti, si) > np.interp(tt, tj, sj)
    ov = ov.assign(persistent=pers)
    first = ~ov.duplicated(["i_vid", "j_vid"])
    net = np.zeros(len(ov), bool)
    for k in np.nonzero(first.to_numpy())[0]:
        ti, si = TI[ov["i_vid"].iat[k]]; tj, sj = TI[ov["j_vid"].iat[k]]
        ta, tb = max(ti[0], tj[0]), min(ti[-1], tj[-1])
        net[k] = (np.interp(ta, ti, si) < np.interp(ta, tj, sj)) and (np.interp(tb, ti, si) > np.interp(tb, tj, sj))
    return ov.assign(net=net)


def block_boot(w, fn, block=BOOT_BLOCK_WIN, n=N_BOOT, seed=0):
    """Estimate and 95% interval of fn(windows) by resampling blocks of consecutive windows."""
    est = fn(w)
    b = (w["win"] // block).to_numpy()
    ub = np.unique(b)
    if len(ub) < 3:
        return est, np.nan, np.nan
    parts = [w[b == k] for k in ub]
    rng = np.random.default_rng(seed)
    vals = [fn(pd.concat([parts[i] for i in rng.integers(0, len(ub), len(ub))])) for _ in range(n)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def fmt_ci(e, lo, hi, nd=2):
    if not np.isfinite(e):
        return "—"
    return f"{e:.{nd}f}" + (f" ({lo:.{nd}f}–{hi:.{nd}f})" if np.isfinite(lo) else "")


def lateral_norm(site, y, y0, y1):
    """0 = median edge when MEDIAN_EDGE is confirmed for the site; otherwise 0 = low-yl edge."""
    z = (np.asarray(y, float) - y0) / (y1 - y0)
    return 1 - z if MEDIAN_EDGE.get(site) == "high" else z


def lateral_label(sites):
    ok = all(MEDIAN_EDGE.get(s) in ("low", "high") for s in sites)
    return ("lateral position (0 = median edge, 1 = curb edge)" if ok
            else "lateral position across the carriageway (0–1)")


def facet_grid(n, ncols=3, h=2.7):
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(W1, h * nrows), squeeze=False)
    for a in axs.flat[n:]:
        a.set_visible(False)
    return fig, list(axs.flat)          # a list, so it can be looped over more than once


def q(s, p):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.quantile(p)) if len(s) else np.nan


# %% 4 · Paper 1 — data audit, cleaning and the cost of skipping it
P1_FIG1_SITES = ["DEL_2", "DEL_4"]        # representative sites for the acceleration histograms
P1_MAP_SITES = DELHI                      # lateral-corridor figure (Noida corridor split is unreliable)
P1_FD_SITES = DELHI                       # flow per metre of width
TRIM_GRID_S = [0, 0.5, 1.0, 1.5, 2.0]
GAP_GRID_M = [0.25, 0.5, 1.0]
SUSTAINED_S = 0.5                         # hard-braking event = at least this long below HARD_BRAKE
SWAP = {"DEL_2": "DEL_3", "DEL_3": "DEL_2"}
PT2_SPEED = {"DEL_2": 33.53, "DEL_3": 11.22, "DEL_4": 34.93, "DEL_5": 36.08, "NDA_1": 33.27, "NDA_2": 28.09}
PT1_DUR = {"DEL_2": 63, "DEL_3": 63, "DEL_4": 66, "DEL_5": 66, "NDA_1": 41, "NDA_2": 41}


def norm_label(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def corridor_bounds(y, gap_m, bin_m=0.25, empty_frac=0.02):
    y = y[np.isfinite(y)]
    if len(y) < 100:
        return []
    lo, hi = np.percentile(y, [0.1, 99.9])
    bins = np.arange(lo, hi + bin_m, bin_m)
    h, _ = np.histogram(y, bins)
    ref = np.median(h[h > 0]) if (h > 0).any() else 0
    empty = h < empty_frac * ref
    out, run = [], 0
    for k, e in enumerate(empty):
        if e:
            run += 1; continue
        if run * bin_m >= gap_m and k - run > 0:
            out.append(float(bins[k - run // 2]))
        run = 0
    return out


def sustained_events(df, dt, thr, min_s):
    """Tracks with at least one run of consecutive samples below thr lasting ≥ min_s, and the run count."""
    d = df[["vid", "t", "lon_acc"]].sort_values(["vid", "t"], kind="mergesort")
    flag = (d["lon_acc"] < thr).to_numpy()
    vid = d["vid"].to_numpy()
    newrun = np.r_[True, (flag[1:] != flag[:-1]) | (vid[1:] != vid[:-1])]
    rid = np.cumsum(newrun)
    runs = pd.DataFrame({"rid": rid[flag], "vid": vid[flag]}).groupby("rid").agg(vid=("vid", "first"), n=("vid", "size"))
    need = int(np.ceil(min_s / dt))
    ev = runs[runs["n"] >= need]
    return ev["vid"].nunique() / d["vid"].nunique(), len(ev)


def raw_position_test(std, dt):
    """Robust SD of lateral acceleration from raw second differences of position (no smoothing), moving rows."""
    d = std[["vid", "t", "yl", "speed"]].sort_values(["vid", "t"], kind="mergesort")
    y, v = d["yl"].to_numpy(), d["vid"].to_numpy()
    same = (v[2:] == v[1:-1]) & (v[1:-1] == v[:-2])
    a = (y[2:] - 2 * y[1:-1] + y[:-2]) / dt ** 2
    a = a[same & (d["speed"].to_numpy()[1:-1] > 5)]
    rsd = 1.4826 * np.median(np.abs(a - np.median(a))) if len(a) else np.nan
    return rsd, rsd * dt ** 2 / np.sqrt(6)       # implied upper bound on white position noise (m)


def workflow_figure(ST):
    sp_lo, sp_hi = ST["spikes_in_last_first_1s"].min(), ST["spikes_in_last_first_1s"].max()
    hz = sorted({round(META[s]["sampling_hz"], 1) for s in SITES})
    steps = [
        ("Load", f"Detect file layout and sampling interval ({' / '.join(str(h) for h in hz)} Hz, not the stated {STATED_FPS} fps)"),
        ("Standardize", "Units, road-aligned axes, class labels ('Tuk-Tuk' → three-wheeler)"),
        ("Directions and U-turns", "Travel direction per track; U-turning tracks flagged"),
        ("Remove", "Static tracks and tracks shorter than 2 s"),
        ("Trim", f"1 s at each track end — removes {sp_lo:.0%}–{sp_hi:.0%} of |a| > {SPIKE:g} m/s² readings"),
        ("Corridors", "Split each direction at near-empty lateral strips (service roads, opposite carriageway)"),
        ("Flag", "Residual outliers, mid-road starts and ends, footprint overlaps"),
        ("Analysis-ready", "Typical vehicle sizes instead of fixed boxes; 10-minute blocks for train/test splits"),
    ]
    fig = plt.figure(figsize=(W1, 6.2), layout=None)
    n = len(steps); top, gap = 0.96, 0.92 / n
    for k, (head, body) in enumerate(steps):
        yc = top - k * gap - gap * 0.38
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (0.03, yc - gap * 0.30), 0.26, gap * 0.60, boxstyle="round,pad=0.006,rounding_size=0.012",
            transform=fig.transFigure, facecolor=PAL[0] if k in (4, 5) else "#eef3fb",
            edgecolor=PAL[0], linewidth=1.0))
        fig.text(0.16, yc, head, ha="center", va="center", fontsize=9, weight="bold",
                 color="white" if k in (4, 5) else INK)
        fig.text(0.32, yc, textwrap.fill(body, 70), ha="left", va="center", fontsize=8.5, color=INK)
        if k < n - 1:
            fig.patches.append(matplotlib.patches.FancyArrowPatch(
                (0.16, yc - gap * 0.31), (0.16, yc - gap * 0.69), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=1.0))
    return fig


GAPS = {}          # site -> list of (start_s, length_s) gaps in the time stamps (used by later parts)
RESETS = {}        # site -> list of (time_s, n_ends, n_starts) instants where tracks are re-initialised


def pack_p1(P):

    # ---------- per-site statistics (raw = released files, clean = cleaned files, all tracks)
    stats, spike_curves, acc_hist, trim_rows, speed_rows, gap_rows, fp_rows = [], {}, {}, [], [], [], []
    bins = np.linspace(-15, 8, 93)
    for s in SITES:
        dt = dt_of(s)
        raw = load_std(s, ["vid", "t", "xl", "yl", "speed", "lon_acc", "cls", "vtype_raw", "L", "W"])
        gaps, within, _ = data_gaps(raw["t"].to_numpy(), raw["vid"].to_numpy(), dt)
        GAPS[s] = gaps
        RESETS[s] = track_resets(raw["t"].to_numpy(), raw["vid"].to_numpy())
        if RESETS[s]:
            print(f"  {s}: track resets at t = " + ", ".join(f"{r[0]:.0f} s ({r[1]} ends/{r[2]} starts)" for r in RESETS[s]))
        if gaps:
            print(f"  {s}: {len(gaps)} gap(s) in the time stamps, longest {max(g[1] for g in gaps):.1f} s at "
                  f"t = {max(gaps, key=lambda g: g[1])[0]:.1f} s")
        cl = load_clean(s, ["vid", "t", "dir", "corridor", "yl", "speed", "lon_acc", "cls", "vtype_raw", "L", "W",
                            "is_nmt", "flag_uturn"])
        g = raw.groupby("vid")["t"]
        t0, t1 = g.transform("min"), g.transform("max")
        tend = np.minimum(raw["t"] - t0, t1 - raw["t"])
        spk = raw["lon_acc"].abs() > SPIKE
        xs = np.linspace(0, 3, 61)
        spike_curves[s] = (xs, [(tend[spk] <= x).mean() for x in xs], [(tend <= x).mean() for x in xs])
        rr = raw["lon_acc"].dropna(); cc = cl["lon_acc"].dropna()
        acc_hist[s] = (np.histogram(rr[(rr >= -15) & (rr <= 8)], bins)[0], np.histogram(cc[(cc >= -15) & (cc <= 8)], bins)[0])
        for T in TRIM_GRID_S:
            keep = ((raw["t"] - t0) >= T) & ((t1 - raw["t"]) >= T)
            trim_rows.append({"Site": s, "Trim (s)": T, "Rows kept (%)": 100 * keep.mean(),
                              "Spikes left (%)": 100 * (spk & keep).sum() / max(spk.sum(), 1)})
        for lab, x in (("as released", raw["speed"]), ("cleaned", cl["speed"])):
            speed_rows.append({"Site": s, "Data": lab, "Mean (km/h)": x.mean(), "p15": x.quantile(0.15),
                               "p50": x.quantile(0.5), "p85": x.quantile(0.85), "SD": x.std()})
        hb_any_raw = raw.loc[raw["lon_acc"] < HARD_BRAKE, "vid"].nunique() / raw["vid"].nunique()
        hb_any_cl = cl.loc[cl["lon_acc"] < HARD_BRAKE, "vid"].nunique() / cl["vid"].nunique()
        hb_sus_raw, ev_raw = sustained_events(raw, dt, HARD_BRAKE, SUSTAINED_S)
        hb_sus_cl, ev_cl = sustained_events(cl, dt, HARD_BRAKE, SUSTAINED_S)
        rsd, sig = raw_position_test(raw, dt)
        labels = cl.groupby("vid")["vtype_raw"].first().map(norm_label)
        stats.append({"site": s, "acc_sd_raw": rr.std(), "acc_sd_clean": cc.std(),
                      "acc_p01_clean": cc.quantile(0.01), "acc_p99_clean": cc.quantile(0.99),
                      "spikes_raw": int(spk.sum()), "spikes_in_last_first_1s": float((tend[spk] <= 1.0).mean()) if spk.any() else np.nan,
                      "hb_any_raw": hb_any_raw, "hb_any_clean": hb_any_cl, "hb_sus_raw": hb_sus_raw, "hb_sus_clean": hb_sus_cl,
                      "tuktuk_share": float((labels == "tuktuk").mean()),
                      "tuktuk_label": ", ".join(sorted(set(cl.loc[cl["vtype_raw"].map(norm_label) == "tuktuk", "vtype_raw"].astype(str).str.strip()))),
                      "lat_acc_raw_rsd": rsd, "pos_noise_bound_cm": 100 * sig,
                      "n_gaps": len(gaps), "gap_total_s": float(sum(g[1] for g in gaps)),
                      "gap_max_s": float(max([g[1] for g in gaps], default=0.0)), "within_track_gap_pct": 100 * within})
        # corridor-threshold sensitivity (main direction, stable motorised tracks)
        trk = cl[(cl["dir"] == 1) & ~cl["is_nmt"]].groupby("vid").agg(y=("yl", "median"), lo=("yl", "min"),
                                                                       hi=("yl", "max"), ut=("flag_uturn", "first"))
        stable = trk[((trk["hi"] - trk["lo"]) < 2.0) & ~trk["ut"]].index
        ys = cl.loc[(cl["dir"] == 1) & cl["vid"].isin(stable), "yl"].to_numpy()
        for gm in GAP_GRID_M:
            b = corridor_bounds(ys, gm)
            band = np.searchsorted(b, trk["y"].to_numpy())
            share = pd.Series(band).value_counts(normalize=True)
            gap_rows.append({"Site": s, "Gap threshold (m)": gm, "Corridors found": len(b) + 1,
                             "Corridors with ≥1% of tracks": int((share >= 0.01).sum()),
                             "Largest corridor share (%)": 100 * share.max()})
        if META[s].get("length_source") == "file":
            per = cl.groupby("vid").agg(cls=("cls", "first"), L=("L", "median"), W=("W", "median"))
            for c in CLASSES:
                pc = per[per["cls"] == c]
                if len(pc):
                    fp_rows.append({"site": s, "cls": c, "nL": pc["L"].round(2).nunique(), "nW": pc["W"].round(2).nunique(),
                                    "L": pc["L"].median(), "W": pc["W"].median()})
        del raw, cl; gc.collect()
    ST = pd.DataFrame(stats).set_index("site")
    print(ST[["tuktuk_share", "tuktuk_label", "lat_acc_raw_rsd", "pos_noise_bound_cm"]])

    # ---------- cost of skipping cleaning (density decomposition, safety indicators)
    macro, micro, fd_pairs = [], [], {}
    for s in SITES:
        dt = dt_of(s)
        allt = load_std(s, ["t", "xl", "speed", "cls"]).rename(columns={"xl": "s"})
        both, _ = edie(allt, 1 / STATED_FPS, dt_true=dt)
        scope_only, _ = edie(allt, dt)
        main = load_clean(s, ["vid", "dir", "corridor", "is_nmt", "t", "s", "yl", "speed", "cls", "L", "W", "frame"],
                          main_only=True)
        clean_fd, core = edie(main, dt)
        fps_only, _ = edie(main, 1 / STATED_FPS, core=core, dt_true=dt)
        width = main["yl"].quantile(0.995) - main["yl"].quantile(0.005)
        fd_pairs[s] = (both, clean_fd, width)
        pk = clean_fd["density_vpkm"].max()
        macro.append({"Site": s,
                      "Main carriageway width (m)": round(float(width), 1),
                      "Peak density, cleaned (veh/km)": int(round(pk)),
                      "Per meter of width (veh/km/m)": round(float(pk / width), 1),
                      f"Rate error only ({STATED_FPS} fps assumed)": int(round(fps_only["density_vpkm"].max())),
                      "Scope error only (all traffic in view)": int(round(scope_only["density_vpkm"].max())),
                      "Both errors (naive use)": int(round(both["density_vpkm"].max()))})
        has_box = META[s].get("length_source") == "file"
        mm = MID[(MID.site == s) & (MID.dir == MAIN_CORRIDOR[s][0]) & (MID.corridor == MAIN_CORRIDOR[s][1])]
        ov_t = CLEAN_SUM.loc[s, "overlap_rows_%_typical_dims"] if s in CLEAN_SUM.index else np.nan
        ov_f = CLEAN_SUM.loc[s, "overlap_rows_%_file_dims"] if s in CLEAN_SUM.index else np.nan
        row = {"Site": s,
               "Acc. SD (m/s²)": f"{ST.loc[s, 'acc_sd_raw']:.2f} → {ST.loc[s, 'acc_sd_clean']:.2f}",
               f"Hard-braking tracks, any reading < {HARD_BRAKE:g} m/s² (%)": f"{100 * ST.loc[s, 'hb_any_raw']:.1f} → {100 * ST.loc[s, 'hb_any_clean']:.1f}",
               f"Hard-braking tracks, ≥ {SUSTAINED_S:g} s run (%)": f"{100 * ST.loc[s, 'hb_sus_raw']:.1f} → {100 * ST.loc[s, 'hb_sus_clean']:.1f}",
               "Rows in a footprint overlap (%)": (f"{ov_f:.2f} → {ov_t:.2f}" if has_box else f"no boxes → {ov_t:.2f}"),
               "Tracks ending mid-road (%)":
                   f"{100 * META[s]['track_integrity']['end_mid_road_share']:.1f} → "
                   + (f"{float(mm['ends_mid_%'].iloc[0]):.1f}" if len(mm) else "n/a")}
        if s in DELHI:
            ld_t, _ = leader_pairs(main, dt, step_s=1.0, dims="typical")
            ttc_t = 100 * (ld_t["ttc_s"] < TTC_CRIT).mean()
            if has_box:
                ld_f, _ = leader_pairs(main, dt, step_s=1.0, dims="file")
                row[f"TTC < {TTC_CRIT:g} s (%)"] = f"{100 * (ld_f['ttc_s'] < TTC_CRIT).mean():.2f} → {ttc_t:.2f}"
            else:
                row[f"TTC < {TTC_CRIT:g} s (%)"] = f"no boxes → {ttc_t:.2f}"
        else:
            row[f"TTC < {TTC_CRIT:g} s (%)"] = "—*"
        micro.append(row)
        del allt, main; gc.collect()
    MACRO, MICRO = pd.DataFrame(macro), pd.DataFrame(micro)

    # ================= MAIN TEXT =================
    P.figure(workflow_figure(ST), "workflow", "Cleaning workflow applied to every file; the two highlighted "
             "steps remove most of the problems found.", main=True)

    rows = []
    for s in SITES:
        m = META[s]
        rows.append({"Site": s, "Rows": f"{m['n_rows']:,}", "Tracks": f"{m['n_vehicles']:,}",
                     "Duration (min)": round(m["duration_min"], 1), "Detected rate (Hz)": round(m["sampling_hz"], 2),
                     "Gaps in time stamps": int(ST.loc[s, "n_gaps"]),
                     "Longest gap (s)": round(ST.loc[s, "gap_max_s"], 1),
                     "Track resets": (", ".join(f"{r[0]:.0f} s" for r in RESETS.get(s, [])) or "none"),
                     "Box columns": "yes" if m.get("length_source") == "file" else "no"})
    P.table(pd.DataFrame(rows), "files", f"Released files as found (documented rate {STATED_FPS} frames per second). "
            "Gaps: intervals with no time stamps longer than 1.5 sampling intervals; Edie windows that straddle a "
            "gap are dropped. Track resets: instants at which many tracks end and new ones start within the same "
            "second (identity re-initialization).", main=True)

    hz = ", ".join(f"{s} {META[s]['sampling_hz']:.2f}" for s in SITES)
    fpd = pd.DataFrame(fp_rows)
    fixed_note = ("One length and one width per class in every file that has them" if len(fpd) and
                  (fpd["nL"].max() == 1 and fpd["nW"].max() == 1) else "Box sizes vary within class in some files")
    bus = fpd[fpd["cls"] == "Bus"]["W"].max() if len(fpd) else np.nan
    issues = [
        ("Sampling rate", f"Stated {STATED_FPS} fps; detected (Hz): {hz}", "all",
         "Detect the interval per file; set windows and steps in seconds"),
        ("Acceleration spikes at track ends",
         f"{ST['spikes_raw'].sum():,} rows with |a| > {SPIKE:g} m/s²; {ST['spikes_in_last_first_1s'].min():.0%}–"
         f"{ST['spikes_in_last_first_1s'].max():.0%} in the first or last 1 s", "all", "Trim 1 s at each track end"),
        ("Acceleration ranges narrower in the dataset paper",
         f"Dataset paper: ranges 1.6–2.3 m/s²; cleaned files, 1st–99th percentile ranges "
         f"{(ST['acc_p99_clean'] - ST['acc_p01_clean']).min():.1f}–{(ST['acc_p99_clean'] - ST['acc_p01_clean']).max():.1f} m/s²",
         "all", "Report any clipping rule; compare like with like"),
        ("Site labels in paper Table 2",
         "; ".join(f"{s} file {META[s]['table2'][0]['speed_obs']:.1f} km/h vs paper {PT2_SPEED[s]} (same label), "
                   f"{PT2_SPEED[SWAP[s]]} (swapped label)" for s in ["DEL_2", "DEL_3"] if s in META),
         "DEL_2, DEL_3", "Use file labels; class mixes confirm them"),
        ("Three-wheeler label",
         f"'{ST['tuktuk_label'].replace('', np.nan).dropna().iloc[0] if (ST['tuktuk_label'] != '').any() else 'Tuk-Tuk'}' "
         f"tracks: {ST['tuktuk_share'].min():.0%}–{ST['tuktuk_share'].max():.0%} of tracks", "all", "Map to M3W"),
        ("Fixed vehicle boxes", f"{fixed_note}; bus box width {bus:.1f} m", ", ".join(sorted(fpd['site'].unique())) if len(fpd) else "none",
         "Use cited typical sizes and a ±10 cm sensitivity test"),
        ("Missing vehicle boxes", "No length/width columns",
         ", ".join(s for s in SITES if META[s].get("length_source") != "file") or "none", "Use cited typical sizes"),
        ("Mixed directions and parallel roads", "; ".join(f"{s}: {len(CORR[CORR.site == s])} corridors" for s in SITES),
         "all", "Split by direction and at near-empty lateral strips"),
        ("Positions smoothed before release",
         "Lateral acceleration from unsmoothed positions (robust SD): " + ", ".join(f"{s} {ST.loc[s, 'lat_acc_raw_rsd']:.2f}" for s in SITES)
         + f" m/s², bounding white position noise at ≤{ST['pos_noise_bound_cm'].max():.1f} cm",
         "all", "State it; do not use the data to test noise robustness"),
    ]
    P.table(pd.DataFrame(issues, columns=["Issue", "Evidence", "Sites", "Cleaning rule"]), "issues",
            "Departures from the documentation and from expected behavior, with the rule that resolves each.", main=True)

    # Fig: spikes vs time from track end (0–3 s, where all the action is)
    fig, a = plt.subplots(figsize=(W2 * 1.6, 3.1))
    for k, s in enumerate(SITES):
        xs, ys, _ = spike_curves[s]
        a.plot(xs, ys, color=PAL[k], lw=1.6, label=s)
    xs, _, allr = spike_curves[SITES[0]]
    a.plot(xs, allr, color=INK2, ls="--", lw=1.2, label=f"all rows ({SITES[0]})")
    a.axvline(1.0, color=INK2, lw=0.8, ls=":")
    a.text(1.05, 0.5, "1 s trim", fontsize=7.5, color=INK2, va="center")
    a.set_xlim(0, 3); a.set_ylim(0, 1.02)
    a.set_xlabel("time to nearest track end (s)"); a.set_ylabel(f"cumulative share of |a| > {SPIKE:g} m/s²")
    a.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    first = min(spike_curves[s][1][0] for s in SITES); firstx = max(spike_curves[s][1][0] for s in SITES)
    P.figure(fig, "spikes_vs_track_end", f"Cumulative share of rows with |a| > {SPIKE:g} m/s² against time to "
             f"the nearest track end, by site ({first:.0%}–{firstx:.0%} lie in the first or last sample); "
             "dashed line = all rows, for reference.", main=True)

    # Fig: lateral corridors (Delhi), common 0.25 m bins, main carriageway filled, others as outlines
    def lateral_panel(a, d):
        bns = np.arange(np.floor(d["yl"].min()), np.ceil(d["yl"].max()) + 0.25, 0.25)
        spec = [("main carriageway", (d["dir"] == 1) & (d["corridor"] == 0), PAL[0], "stepfilled", "-"),
                ("other same-direction corridors", (d["dir"] == 1) & (d["corridor"] != 0), PAL[1], "step", "-"),
                ("opposite direction", d["dir"] == -1, PAL[2], "step", "--")]
        for lab, m, col, ht, ls in spec:
            if m.any():
                a.hist(d.loc[m, "yl"], bins=bns, color=col, label=lab, histtype=ht, ls=ls,
                       lw=0 if ht == "stepfilled" else 1.4, alpha=0.9 if ht == "stepfilled" else 1)
        a.set_yscale("log"); a.set_xlabel("lateral position (m)")

    def legend_below(fig, axs, ncol=3):
        hl = {}
        for a in axs:
            for hh, ll in zip(*a.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=ncol)

    msites = [s for s in P1_MAP_SITES if s in SITES]
    fig, axs = facet_grid(len(msites), ncols=2, h=2.6)
    for a, s in zip(axs, msites):
        d = load_clean(s, ["dir", "corridor", "yl", "is_nmt"]); d = d[~d["is_nmt"]]
        d = d.sample(min(len(d), 400_000), random_state=0)
        lateral_panel(a, d); a.set_title(s, loc="left"); a.set_ylabel("rows (log)")
    legend_below(fig, axs)
    P.figure(fig, "lateral_corridors", "Lateral occupancy by direction and corridor at the Delhi sites "
             "(0.25 m bins): service roads, curbside lanes and the opposite carriageway sit inside single files. "
             f"(Noida sites: {P.ref('F', 'lateral_corridors_noida')}.)", main=True)

    # Fig: footprints
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 3.1))
    if len(fpd):
        s0 = fpd["site"].iloc[0]
        f0 = fpd[fpd["site"] == s0].set_index("cls").reindex([c for c in CLASSES if c in set(fpd["cls"])])
        x = np.arange(len(f0))
        a1.bar(x - 0.2, f0["W"], 0.38, color=PAL[1], label="file box")
        a1.bar(x + 0.2, [TYPICAL_DIMS[c][1] for c in f0.index], 0.38, color=PAL[0], label=TYPICAL_LABEL)
        a1.set_xticks(x); a1.set_xticklabels(f0.index); a1.set_ylabel("width (m)")
        a1.set_title(f"(a) widths, {s0}", loc="left"); a1.legend(loc="upper left")
    cs = CLEAN_SUM.reindex(SITES); x = np.arange(len(SITES))
    hasbox = np.array([META[s].get("length_source") == "file" for s in SITES])
    fv = np.where(hasbox, cs["overlap_rows_%_file_dims"].to_numpy(float), np.nan)
    a2.bar(x[hasbox] - 0.2, fv[hasbox], 0.38, color=PAL[1], label="file boxes")
    a2.bar(x + 0.2, cs["overlap_rows_%_typical_dims"], 0.38, color=PAL[0], label=TYPICAL_LABEL.replace("typical", "typical boxes"))
    top = np.nanmax(np.r_[fv, cs["overlap_rows_%_typical_dims"].to_numpy(float)])
    for k in np.where(~hasbox)[0]:
        a2.text(x[k] - 0.2, top * 0.03, "no\nboxes", ha="center", va="bottom", fontsize=6.5, color=INK2)
    a2.set_xticks(x); a2.set_xticklabels(SITES, rotation=30); a2.set_ylabel("rows in a footprint overlap (%)")
    a2.set_title("(b) footprint overlaps", loc="left"); a2.legend(loc="upper right")
    P.figure(fig, "footprints", "Fixed vehicle boxes in the released files: (a) width per class against typical "
             f"sizes ({TYPICAL_DIMS_SOURCE}); (b) share of rows whose box overlaps another vehicle. DEL_4 and DEL_5 "
             "carry no boxes, so only typical sizes apply there.", main=False)

    # Fig: FD naive vs cleaned (congested site) + cleaned Delhi sites per metre of width
    key = "DEL_2" if "DEL_2" in fd_pairs else SITES[0]
    nv, cf, _ = fd_pairs[key]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 3.2))
    a.scatter(nv["density_vpkm"], nv["speed_kmh"], s=14, color=INK2, marker="x", lw=1, label=f"naive ({STATED_FPS} fps, all traffic)")
    a.scatter(cf["density_vpkm"], cf["speed_kmh"], s=14, color=PAL[0], marker="o", label="cleaned (detected rate,\nmain carriageway)")
    a.set_xlabel("density (veh/km)"); a.set_ylabel("space-mean speed (km/h)"); a.legend(fontsize=7, loc="best")
    a.set_title(f"(a) {key}: naive against cleaned", loc="left")
    mk = ["o", "s", "^", "D", "v", "P"]
    for k, s in enumerate([s for s in DELHI if s in fd_pairs]):
        c2, w = fd_pairs[s][1], fd_pairs[s][2]
        b.scatter(c2["density_vpkm"] / w, c2["speed_kmh"], s=12, color=PAL[k], marker=mk[k], label=f"{s} ({w:.1f} m)")
    b.set_xlabel("density per meter of width (veh/km/m)"); b.set_ylabel("space-mean speed (km/h)")
    b.set_title("(b) cleaned Delhi sites", loc="left"); b.legend(fontsize=7, loc="best")
    P.figure(fig, "fd_naive_vs_clean", f"Speed–density relations from {WINDOW_S} s Edie windows: (a) naive use of "
             f"the released {key} file against the cleaned main carriageway; (b) cleaned main carriageways of the "
             "Delhi sites, density divided by carriageway width so that sites of different width can be compared.", main=True)

    P.table(MACRO, "cost_macroscopic", "How naive use of the released files changes peak density (main carriageway, "
            f"{WINDOW_S} s Edie windows). The naive value combines a frame-rate error ({STATED_FPS} fps assumed instead "
            "of the detected rate) and a scope error (all traffic in view instead of one carriageway).", digits=1, main=True)
    P.table(MICRO, "cost_microscopic", "How naive use of the released files changes microscopic indicators "
            "(released → cleaned). Hard braking: tracks with any reading below "
            f"{HARD_BRAKE:g} m/s², or with a run of at least {SUSTAINED_S:g} s below it. Footprint overlap and TTC: file "
            "boxes → typical sizes. TTC: nearest leader with lateral overlap, sampled every 1 s. *Not computed for the "
            f"Noida sites because their corridor split is unreliable ({P.ref('F', 'lateral_corridors_noida')}).", main=True)

    # ================= SUPPLEMENTARY =================

    rows = []
    for s in SITES:
        m = META[s]; comp = pd.DataFrame(m["composition"])
        rows.append({"Site": s, "Paper duration (min)": PT1_DUR.get(s), "File duration (min)": m["duration_min"],
                     "File mean speed (km/h)": m["table2"][0]["speed_obs"],
                     "Paper, same label (km/h)": PT2_SPEED.get(s),
                     "Paper, swapped label (km/h)": PT2_SPEED.get(SWAP[s]) if s in SWAP else np.nan,
                     "Max class-share gap (pp)": float(comp["diff_pp"].abs().max()) if "diff_pp" in comp else np.nan})
    P.table(pd.DataFrame(rows), "documentation_vs_data", "Dataset paper (Ali et al., arXiv 2512.11898, Tables 1–2) "
            "against the released files. Class shares use the three-wheeler mapping.", main=False)

    lg = CLEAN_LOG[CLEAN_LOG.site.isin(SITES)].copy()
    lg["k"] = lg.groupby("site").cumcount()
    base = lg[lg["k"] == 0].set_index("site")
    rows = []
    for s, g in lg.groupby("site", sort=False):
        r = {"Site": s, "Tracks loaded": int(base.loc[s, "tracks"]), "Rows loaded": int(base.loc[s, "rows"])}
        for _, x in g.iloc[1:].iterrows():
            if x["step"].startswith("final"):
                continue
            short = x["step"].split("·")[-1].strip()
            r[f"{short}: tracks kept (%)"] = 100 * x["tracks"] / base.loc[s, "tracks"]
            r[f"{short}: rows kept (%)"] = 100 * x["rows"] / base.loc[s, "rows"]
        rows.append(r)
    P.table(pd.DataFrame(rows), "cleaning_log", "Share of tracks and rows kept after each removal step.", main=False)

    if len(fpd):
        rows = []
        for c in [c for c in CLASSES if c in set(fpd["cls"])]:
            f = fpd[fpd["cls"] == c]
            same = (f["L"].round(2).nunique() == 1) and (f["W"].round(2).nunique() == 1) and f["nL"].max() == 1
            rows.append({"Class": c, "File length (m)": f["L"].iloc[0], "File width (m)": f["W"].iloc[0],
                         "Typical length (m)": TYPICAL_DIMS[c][0], "Typical width (m)": TYPICAL_DIMS[c][1],
                         "Files": len(f), "Identical in every file, one value per class": "yes" if same else "no"})
        P.table(pd.DataFrame(rows), "footprints", f"Vehicle box sizes in the released files against typical sizes ({TYPICAL_DIMS_SOURCE}).", main=False)

    P.table(pd.DataFrame(trim_rows), "trim_sensitivity", "Trim-length sensitivity: share of rows kept and of "
            f"|a| > {SPIKE:g} m/s² spikes left after trimming T seconds at each track end.", main=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 3.1))
    TR = pd.DataFrame(trim_rows)
    for k, s in enumerate(SITES):
        t = TR[TR["Site"] == s]
        a1.plot(t["Trim (s)"], t["Spikes left (%)"], color=PAL[k], marker="o", ms=4, label=s)
        a2.plot(t["Trim (s)"], t["Rows kept (%)"], color=PAL[k], marker="o", ms=4, label=s)
    a1.set_xlabel("trim at each end (s)"); a1.set_ylabel("spikes left (%)"); a1.set_title("(a) spikes left", loc="left")
    a2.set_xlabel("trim at each end (s)"); a2.set_ylabel("rows kept (%)"); a2.set_title("(b) data kept", loc="left")
    a1.legend(fontsize=7, ncol=2)
    P.figure(fig, "trim_sensitivity", "Choosing the trim length: spikes removed against data kept.", main=False)

    P.table(pd.DataFrame(speed_rows), "speed_before_after", "Speed distributions before and after cleaning (all "
            "tracks): trimming removes acceleration artifacts without shifting speeds.", main=False)
    P.table(pd.DataFrame(gap_rows), "corridor_gap_sensitivity", "Sensitivity of the corridor split to the width "
            "of the near-empty strip (main direction, stable motorized tracks).", main=False)

    # Fig S: acceleration histograms (representative sites, zero bins masked, no clipping pile-up)
    fs = [s for s in P1_FIG1_SITES if s in SITES]
    fig, axs = plt.subplots(1, len(fs), figsize=(W1, 3.0), squeeze=False)
    ctr = (bins[:-1] + bins[1:]) / 2
    for a, s in zip(axs[0], fs):
        r, c = acc_hist[s]
        a.step(ctr, np.where(r > 0, r, np.nan), where="mid", color=INK2, lw=1, label="as released")
        a.step(ctr, np.where(c > 0, c, np.nan), where="mid", color=PAL[0], lw=1.5, label="cleaned")
        a.set_yscale("log"); a.set_title(s, loc="left"); a.set_xlabel("longitudinal acc. (m/s²)")
    axs[0][0].set_ylabel("rows (log)"); axs[0][0].legend(loc="upper left")
    P.figure(fig, "acc_raw_vs_clean", "Reported longitudinal acceleration before and after trimming 1 s at each "
             "track end (rows within −15…8 m/s²; empty bins not drawn).", main=False)

    nsites = [s for s in SITES if s not in P1_MAP_SITES]
    if nsites:
        fig, axs = facet_grid(len(nsites), ncols=2, h=2.6)
        for a, s in zip(axs, nsites):
            d = load_clean(s, ["dir", "corridor", "yl", "is_nmt"]); d = d[~d["is_nmt"]]
            d = d.sample(min(len(d), 400_000), random_state=0)
            lateral_panel(a, d); a.set_title(s, loc="left"); a.set_ylabel("rows (log)")
        legend_below(fig, axs)
        P.figure(fig, "lateral_corridors_noida", "Lateral occupancy at the Noida sites: crossing and turning traffic "
                 "fills the strips between carriageways, so the corridor split is unreliable there.", main=False)

    fsites = [s for s in P1_FD_SITES if s in fd_pairs]
    fig, axs = facet_grid(len(fsites), ncols=2, h=2.6)
    for a, s in zip(axs, fsites):
        cf, w = fd_pairs[s][1], fd_pairs[s][2]
        a.scatter(cf["density_vpkm"] / w, cf["flow_vph"] / w, s=8, color=PAL[0])
        a.set_title(f"{s} (width {w:.1f} m)", loc="left"); a.set_xlabel("density (veh/km per m width)")
        a.set_ylabel("flow (veh/h per m width)")
    P.figure(fig, "fd_per_metre", "Flow–density relations of the cleaned Delhi main carriageways, per meter of "
             "carriageway width (30 s Edie windows).", main=False)

    fig, a = plt.subplots(figsize=(W2 * 1.5, 3.0))
    y = np.arange(len(SITES))
    obs = [META[s]["table2"][0]["speed_obs"] for s in SITES]
    a.scatter(obs, y, color=PAL[0], s=36, zorder=3, label="file (as released)")
    a.scatter([PT2_SPEED.get(s, np.nan) for s in SITES], y, facecolors="none", edgecolors=PAL[1], s=36, lw=1.5,
              zorder=3, label="paper, same label")
    sw = [PT2_SPEED[SWAP[s]] if s in SWAP else np.nan for s in SITES]
    a.scatter(sw, y, marker="D", color=PAL[2], s=30, zorder=3, label="paper, swapped label (DEL_2↔DEL_3)")
    for k in range(len(SITES)):
        a.plot([obs[k], PT2_SPEED.get(SITES[k], np.nan)], [k, k], color=GRID, lw=1.5, zorder=1)
    a.set_yticks(y); a.set_yticklabels(SITES); a.set_xlabel("mean speed (km/h)"); a.legend(fontsize=7, loc="lower left")
    P.figure(fig, "table2_label_check", "Mean speed in each released file against the dataset paper's Table 2 "
             "value for the same label and, for DEL_2 and DEL_3, the swapped label.", main=False)

    fig, a = plt.subplots(figsize=(W2 * 1.5, 3.0))
    x = np.arange(len(SITES))
    released = [100 * META[s]["track_integrity"]["end_mid_road_share"] for s in SITES]
    clu, spr = [], []
    for s in SITES:
        mm = MID[(MID.site == s) & (MID.dir == 1) & (MID.corridor == 0)]
        e = float(mm["ends_mid_%"].iloc[0]) if len(mm) else np.nan
        sh = float(mm["ends_densest10m_share"].iloc[0]) if len(mm) and pd.notna(mm["ends_densest10m_share"].iloc[0]) else 0.0
        clu.append(e * sh); spr.append(e * (1 - sh))
    a.bar(x - 0.2, released, 0.38, color=INK2, label="as released, all traffic")
    a.bar(x + 0.2, clu, 0.38, color=PAL[1], label="main carriageway: at one access point (densest 10 m)")
    a.bar(x + 0.2, spr, 0.38, bottom=clu, color=PAL[0], label="main carriageway: spread out")
    a.set_xticks(x); a.set_xticklabels(SITES, rotation=30); a.set_ylabel("tracks ending mid-road (%)"); a.legend(fontsize=7)
    P.figure(fig, "mid_road_ends", "Apparent track loss before and after separating directions and parallel roads; "
             "endings bunched within one 10 m stretch mark a real exit, not lost tracks.", main=False)


# %% 5 · Part 2 — two-wheeler filtering in congestion (main text) + details (supplement)
REGIME_COL = dict(zip([r[2] for r in REGIMES], [PAL[7], PAL[3], PAL[2], PAL[0]]))


def filtering_inputs(d, dt, widths="typical"):
    """Edie windows by class, classified passing events, and per-window pass counts for one data set."""
    win, core = edie(d, dt, by_class=True)
    ov = overtakes(d, dt, core, step_s=PASS_STEP_S, widths=widths)
    if len(ov):
        ov = classify_passes(ov, track_index(d))
        ov["win"] = (ov["t"] // WINDOW_S).astype(int)
        ov = ov.merge(win[["win", "regime", "density_vpkm"]], on="win", how="inner")
        ov["close"] = ov["persistent"] & (ov["clearance_m"] < NEAR_M)
        m2 = ov[ov["passer_cls"] == "M2W"]
        cnt = pd.DataFrame({"gross_M2W": m2.groupby("win").size(),
                            "pers_M2W": m2[m2["persistent"]].groupby("win").size(),
                            "net_M2W": m2[m2["net"]].groupby("win").size(),
                            "close_M2W": m2[m2["close"]].groupby("win").size()})
        win = win.merge(cnt, left_on="win", right_index=True, how="left")
    for c in ["gross_M2W", "pers_M2W", "net_M2W", "close_M2W"]:
        win[c] = win[c].fillna(0) if c in win else 0.0
    return win, core, ov


def ratio(num, den):
    return lambda w: w[num].sum() / w[den].sum() if w[den].sum() > 0 else np.nan


def rel_speed(w):            # two-wheeler space-mean speed ÷ stream space-mean speed = flow share ÷ presence share
    return (w["flow_M2W"].sum() / w["flow_vph"].sum()) / (w["dens_M2W"].sum() / w["density_vpkm"].sum())


def m2w_car(w):
    return (w["flow_M2W"].sum() / w["dens_M2W"].sum()) / (w["flow_Car"].sum() / w["dens_Car"].sum())


def pack_p4(P):
    order = [r[2] for r in REGIMES]
    res, naive = {}, {}
    for role, s in P4_SITES.items():
        if s not in SITES:
            continue
        dt = dt_of(s)
        d = load_clean(s, ["vid", "dir", "corridor", "is_nmt", "frame", "t", "s", "yl", "speed", "cls"], main_only=True)
        d = d[d["cls"].isin(CLASSES)]
        win, core, ov = filtering_inputs(d, dt)
        y0, y1 = np.nanpercentile(d["yl"], [0.5, 99.5])
        ym = d.groupby("vid").agg(cls=("cls", "first"), y=("yl", "median"))
        ym["y_norm"] = lateral_norm(s, ym["y"], y0, y1)
        res[s] = dict(win=win, ov=ov, ym=ym, n=d.groupby("cls")["vid"].nunique(),
                      hours=(d["t"].max() - d["t"].min()) / 3600)
        if role == "congested":
            cw = win[win["regime"] == order[0]]
            t0 = (cw["win"].median() * WINDOW_S) if len(cw) else d["t"].min()
            res[s]["ts"] = d[(d["t"] >= t0 - 90) & (d["t"] < t0 + 90)]
        del d; gc.collect()
        # naive: released file, main travel direction only (the one filter any user applies), untrimmed,
        # all corridors, released boxes where the file has them
        raw = load_std(s, ["vid", "dir", "frame", "t", "xl", "yl", "speed", "cls", "L", "W"])
        raw = raw[(raw["dir"] == 1) & raw["cls"].isin(CLASSES)].rename(columns={"xl": "s"})
        wn = "file" if META[s].get("length_source") == "file" else "typical"
        naive[s] = filtering_inputs(raw, dt, widths=wn)[0]
        del raw; gc.collect()
    if not res:
        print("filtering part: no sites"); return

    # ---------- MAIN TABLE: filtering measures by regime, cleaned (95% block-bootstrap CI) vs naive
    rows = []
    for s, r in res.items():
        for o in order:
            w = r["win"][r["win"]["regime"] == o]
            if len(w) < 3:
                continue
            wn = naive[s][naive[s]["regime"] == o]
            rows.append({
                "Site": s, "Regime (km/h)": o, "Windows": len(w),
                "M2W share of vehicles present (%)": round(100 * w["dens_M2W"].sum() / w["density_vpkm"].sum(), 1),
                "M2W ÷ stream speed": fmt_ci(*block_boot(w, rel_speed)),
                "M2W ÷ car speed": fmt_ci(*block_boot(w, m2w_car)),
                f"Close passes by M2W per M2W veh-km (< {NEAR_M:g} m, held {PERSIST_S:g} s)":
                    fmt_ci(*block_boot(w, ratio("close_M2W", "km_M2W")), nd=1),
                "Naive: M2W ÷ stream speed": f"{rel_speed(wn):.2f}" if len(wn) else "—",
                "Naive: close passes per veh-km": f"{wn['close_M2W'].sum() / wn['km_M2W'].sum():.1f}" if len(wn) else "—",
            })
    P.table(pd.DataFrame(rows), "filtering_measures",
            "Two-wheeler (M2W) filtering measures by traffic regime on the cleaned main carriageway, with 95% "
            f"block-bootstrap intervals (blocks of {BOOT_BLOCK_WIN} consecutive 30 s windows), against naive use of "
            "the released file (main travel direction, all corridors, untrimmed tracks, released boxes). "
            "M2W ÷ stream speed equals the M2W share of flow divided by the M2W share of vehicles present.", main=True)

    # ---------- MAIN FIGURES
    cs = P4_SITES["congested"]
    if cs in res and "ts" in res[cs]:
        ts = res[cs]["ts"]
        fig, a = plt.subplots(figsize=(W1, 3.3))
        for grp, col, lw, lab in [(ts[ts["cls"] != "M2W"], "#9a9893", 0.5, "other vehicles"),
                                  (ts[ts["cls"] == "M2W"], PAL[1], 0.8, "two-wheelers")]:
            segs = []
            for _, g in grp.groupby("vid"):
                tt, ss = g["t"].to_numpy(), g["s"].to_numpy()
                br = np.nonzero(np.diff(tt) > 1.5 * dt_of(cs))[0] + 1          # do not draw across data gaps
                segs += [np.c_[a_, b_] for a_, b_ in zip(np.split(tt, br), np.split(ss, br)) if len(a_) > 1]
            a.add_collection(LineCollection(segs, colors=col, linewidths=lw))
            a.plot([], [], color=col, lw=2, label=lab)
        for g0, gl in GAPS.get(cs, []):
            if ts["t"].min() <= g0 <= ts["t"].max():
                a.axvspan(g0, g0 + gl, color="#d9d8d4", lw=0, label="no data")
        for r0 in RESETS.get(cs, []):
            if ts["t"].min() <= r0[0] <= ts["t"].max():
                a.axvline(r0[0] + 0.5, color=INK, lw=0.9, ls=":", label="track reset")
        a.autoscale(); a.set_xlabel("time (s)"); a.set_ylabel("distance along road (m)")
        hl = dict(zip(*a.get_legend_handles_labels()[::-1]))
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside upper center", ncol=3)
        P.figure(fig, "time_space", f"Time–space diagram of the {cs} main carriageway, 3-minute excerpt centered "
                 "in the congested period: two-wheelers (orange) cut through the stop-and-go waves formed by other "
                 "vehicles.", main=True)

    fig, a = plt.subplots(figsize=(W2 * 1.6, 3.2))
    mk = {s: m for s, m in zip(res, ["o", "^"])}
    for s, r in res.items():
        w = r["win"]
        for o in order:
            ww = w[w["regime"] == o]
            if len(ww):
                a.scatter(100 * ww["dens_M2W"] / ww["density_vpkm"], 100 * ww["flow_M2W"] / ww["flow_vph"], s=14,
                          color=REGIME_COL[o], marker=mk[s], alpha=0.85, label=f"{s}, {o} km/h")
    lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
    a.plot(lim, lim, color=INK2, ls="--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("two-wheeler share of vehicles present (%)"); a.set_ylabel("two-wheeler share of flow (%)")
    a.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    P.figure(fig, "m2w_share", "Per 30 s window: two-wheelers' share of flow against their share of vehicles "
             "present, colored by traffic regime; points above the dashed line mean that two-wheelers move faster "
             "than the stream.", main=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 3.2))
    for k, (s, r) in enumerate(res.items()):
        w = r["win"]
        w = w[w["km_M2W"] > 0.05]
        bins = np.unique(np.quantile(w["density_vpkm"], np.linspace(0, 1, 9)))
        g = w.groupby(pd.cut(w["density_vpkm"], bins, include_lowest=True), observed=True)
        x = g["density_vpkm"].median()
        a1.plot(x, g["close_M2W"].sum() / g["km_M2W"].sum(), color=PAL[k], marker=mk[s], ms=5, label=f"{s}, close and held")
        a1.plot(x, g["gross_M2W"].sum() / g["km_M2W"].sum(), color=PAL[k], ls=":", lw=1.2, label=f"{s}, all order changes")
    a1.set_xlabel("density (veh/km)"); a1.set_ylabel("passes by two-wheelers per veh-km")
    a1.set_title("(a) passing rate", loc="left"); a1.legend(fontsize=6.5)
    r = res[cs]["ov"]
    m = r[(r["passer_cls"] == "M2W") & r["persistent"]] if len(r) else r
    xs = np.linspace(-0.5, 3.0, 141)
    for lab, mask, col in [("window speed < 25 km/h", m["regime"].isin(order[:2]), PAL[1]),
                           ("window speed ≥ 25 km/h", m["regime"].isin(order[2:]), PAL[0])]:
        c = np.sort(m.loc[mask, "clearance_m"].to_numpy())
        if len(c) < 20:
            continue
        e = lambda sh: np.searchsorted(c + sh, xs, side="right") / len(c)
        a2.fill_between(xs, e(-WIDTH_SENS_M), e(WIDTH_SENS_M), color=col, alpha=0.18, lw=0)
        a2.plot(xs, e(0.0), color=col, lw=1.8, label=lab)
    a2.axvline(0, color=INK2, lw=0.8, ls=":"); a2.axvline(NEAR_M, color=INK2, lw=0.8, ls="--")
    a2.set_xlabel("lateral clearance at the pass (m)"); a2.set_ylabel("cumulative share of passes")
    a2.set_title(f"(b) clearance, {cs}", loc="left"); a2.legend(fontsize=6.5, loc="lower right")
    P.figure(fig, "passing", f"Two-wheeler passing: (a) passes per two-wheeler vehicle-kilometer against density "
             f"(density octiles), counting close passes (clearance below {NEAR_M:g} m, new order held for "
             f"{PERSIST_S:g} s) and all order changes; (b) cumulative distribution of lateral clearance at held "
             f"passes at {cs} (typical widths; band = every width ±{100 * WIDTH_SENS_M:.0f} cm; dashed line = "
             f"close-pass limit).", main=True)

    # ---------- SUPPLEMENT
    rows = []
    for s, r in res.items():
        wc = r["win"]["regime"].value_counts()
        rows.append({"Site": s, "Hours": round(r["hours"], 2), **{f"{c} tracks": int(r["n"].get(c, 0)) for c in CLASSES},
                     **{f"Windows {o.split()[0]}": int(wc.get(o, 0)) for o in order}})
    P.table(pd.DataFrame(rows), "sample", "Filtering analysis sample: tracks on the main carriageway and 30 s "
            "windows per traffic regime.", main=False)

    rows = []
    for s, r in res.items():
        for o in order:
            w = r["win"][r["win"]["regime"] == o]
            if len(w):
                rows.append({"Site": s, "Regime (km/h)": o, "Windows": len(w),
                             **{f"{c} (km/h)": round(w[f"flow_{c}"].sum() / w[f"dens_{c}"].sum(), 1)
                                for c in CLASSES if f"flow_{c}" in w and w[f"dens_{c}"].sum() > 0}})
    P.table(pd.DataFrame(rows), "speed_by_class", "Space-mean speed of each class by regime (ratio of sums over "
            "windows).", main=False)

    rows = []
    for s, r in res.items():
        ov, w = r["ov"], r["win"]
        if not len(ov):
            continue
        for o in order:
            ww = w[w["regime"] == o]
            if not len(ww):
                continue
            row = {"Site": s, "Regime (km/h)": o}
            for c in ["M2W", "Car", "M3W"]:
                km = ww[f"km_{c}"].sum() if f"km_{c}" in ww else 0
                oc = ov[(ov["regime"] == o) & (ov["passer_cls"] == c)]
                for lab, n in [("all", len(oc)), ("held", int(oc["persistent"].sum())), ("net", int(oc["net"].sum())),
                               ("close", int(oc["close"].sum()))]:
                    row[f"{c} {lab} per veh-km"] = round(n / km, 1) if km > 0 else np.nan
            rows.append(row)
    P.table(pd.DataFrame(rows), "passing_detail", "Passing rates by passing class and regime under four "
            f"definitions: all order changes between samples {PASS_STEP_S:g} s apart; held (new order still holds "
            f"{PERSIST_S:g} s later); net (pair's order reversed over their common presence); close (held and "
            f"clearance below {NEAR_M:g} m). Per vehicle-kilometer traveled by the passing class.", main=False)

    rows = []
    for s, r in res.items():
        ov = r["ov"]
        if not len(ov):
            continue
        m = ov[(ov["passer_cls"] == "M2W") & ov["persistent"]]
        for o in order:
            for c in CLASSES:
                x = m[(m["regime"] == o) & (m["passed_cls"] == c)]["clearance_m"]
                if len(x) >= 20:
                    rows.append({"Site": s, "Regime (km/h)": o, "Passed class": c, "Held passes": len(x),
                                 "Close passes (%)": round(100 * (x < NEAR_M).mean(), 1),
                                 "Median clearance (m)": round(x.median(), 2), "p05 (m)": round(x.quantile(0.05), 2),
                                 "Share < 0.3 m (%)": round(100 * (x < 0.3).mean(), 2)})
    P.table(pd.DataFrame(rows), "clearance_close_passes", "Lateral clearance at held two-wheeler passes by passed "
            "class and regime (typical widths; groups with at least 20 passes).", main=False)

    rows = []
    for s, r in res.items():
        ov = r["ov"]
        if not len(ov):
            continue
        m = ov[(ov["passer_cls"] == "M2W") & ov["persistent"]]
        for dW, lab in [(-WIDTH_SENS_M, "−10 cm"), (0, "typical"), (WIDTH_SENS_M, "+10 cm")]:
            clr = m["offset_m"] - (m["half_widths_m"] + dW)
            rows.append({"Site": s, "Vehicle widths": lab, "Median clearance (m)": round(clr.median(), 2),
                         f"Close (< {NEAR_M:g} m) (%)": round(100 * (clr < NEAR_M).mean(), 1),
                         "Share < 0.3 m (%)": round(100 * (clr < 0.3).mean(), 2), "Share < 0 m (%)": round(100 * (clr < 0).mean(), 2)})
    P.table(pd.DataFrame(rows), "width_sensitivity", "Sensitivity of held two-wheeler passing clearance to vehicle "
            "widths (every vehicle ±10 cm).", main=False)

    A = {c: TYPICAL_DIMS[c][0] * TYPICAL_DIMS[c][1] for c in CLASSES}
    rows = []
    for s, r in res.items():
        w = r["win"]
        for o in order:
            ww = w[w["regime"] == o]
            if not len(ww) or "flow_Car" not in ww:
                continue
            vcar = ww["flow_Car"].sum() / ww["dens_Car"].sum()
            row = {"Site": s, "Regime (km/h)": o}
            for c in CLASSES[1:]:
                if f"flow_{c}" in ww and ww[f"dens_{c}"].sum() > 0:
                    vi = ww[f"flow_{c}"].sum() / ww[f"dens_{c}"].sum()
                    row[f"PCU {c}"] = round((vcar / vi) / (A["Car"] / A[c]), 2)
            rows.append(row)
    P.table(pd.DataFrame(rows), "dynamic_pcu", "Dynamic passenger-car units by regime, PCU = (V_car / V_i) / "
            f"(A_car / A_i), with typical projected areas ({TYPICAL_DIMS_SOURCE}).", main=False)

    fig, axs = plt.subplots(1, len(res), figsize=(W1, 3.1), sharey=True, squeeze=False)
    for a, (s, r) in zip(axs[0], res.items()):
        w = r["win"]
        bins = np.unique(np.quantile(w["density_vpkm"], np.linspace(0, 1, 9)))
        w = w.assign(b=pd.cut(w["density_vpkm"], bins, include_lowest=True))
        for c in ["Car", "M2W", "M3W"]:
            if f"speed_{c}" not in w:
                continue
            g = w.groupby("b", observed=True)
            xm, ym_ = g["density_vpkm"].median(), g[f"speed_{c}"].median()
            lo_, hi_ = g[f"speed_{c}"].quantile(0.25), g[f"speed_{c}"].quantile(0.75)
            a.errorbar(xm, ym_, yerr=[ym_ - lo_, hi_ - ym_], color=CLS_COLOR[c], marker=CLS_MARK[c], ms=4,
                       lw=1.5, capsize=2, label=c)
        a.set_title(s, loc="left"); a.set_xlabel("density (veh/km)")
    axs[0][0].set_ylabel("class space-mean speed (km/h)"); axs[0][0].legend()
    P.figure(fig, "speed_density_by_class", "Class space-mean speed against density (median and interquartile "
             "range within density octiles).", main=False)


# %% 6 · Supplement — calibration-ready targets for simulation (corrected definitions)
def pack_p2(P):
    sites = [s for s in P2_SITES if s in SITES]
    R = {}
    for s in sites:
        dt = dt_of(s)
        d = load_clean(s, ["vid", "dir", "corridor", "is_nmt", "block", "frame", "t", "s", "yl", "speed", "lon_acc",
                           "flag_acc", "cls"], main_only=True)
        d = d[d["cls"].isin(CLASSES)]
        win, core = edie(d, dt, by_class=True)
        ld, sd = leader_pairs(d, dt, step_s=0.5, dims="typical")
        ld["win"] = (ld["t"] // WINDOW_S).astype(int)
        ld = ld.merge(win[["win", "regime", "speed_kmh"]], on="win", how="left")
        step = max(1, int(round(0.5 / dt)))
        samp = d[d["frame"] % step == 0][["t", "vid", "speed", "cls"]].copy()
        samp["win"] = (samp["t"] // WINDOW_S).astype(int)
        samp = samp.merge(win[["win", "speed_kmh"]], on="win", how="inner")
        followed = set(zip(np.round(ld["t"], 3), ld["f_vid"].astype(np.int64)))
        key = list(zip(np.round(samp["t"], 3), samp["vid"].astype(np.int64)))
        leaderfree = samp[[k not in followed for k in key]]
        y0, y1 = np.nanpercentile(d["yl"], [0.5, 99.5])
        trk = d.groupby("vid").agg(cls=("cls", "first"), t0=("t", "first"), y=("yl", "median"))
        trk["y_norm"] = lateral_norm(s, trk["y"], y0, y1)
        acc = d.loc[~d["flag_acc"], ["cls", "speed", "lon_acc"]]
        R[s] = dict(win=win, core=core, ld=ld, sd=sd, lf=leaderfree, trk=trk, width=y1 - y0, acc=acc,
                    tmax=d["t"].max(), occ=d[["s", "yl"]].sample(n=min(len(d), 300_000), random_state=0))
        del d; gc.collect()
    if not sites:
        return

    rows = []
    for s in sites:
        mm = MID[(MID.site == s) & (MID.dir == 1) & (MID.corridor == 0)]
        m = mm.iloc[0] if len(mm) else {}
        rows.append({"Site": s, "Carriageway width (m)": round(R[s]["width"], 2),
                     "Core stretch (m)": round(R[s]["core"][1] - R[s]["core"][0], 1),
                     "Edge drift corrected (m per 100 m)": round(STRAIGHT.get(s, {}).get("drift_m_per_100m", np.nan), 2),
                     "Mid-road starts (%)": m.get("starts_mid_%"), "Start cluster at (m)": m.get("starts_cluster_at_s_m"),
                     "Share in densest 10 m (starts)": m.get("starts_densest10m_share"),
                     "Mid-road ends (%)": m.get("ends_mid_%"), "End cluster at (m)": m.get("ends_cluster_at_s_m"),
                     "Share in densest 10 m (ends)": m.get("ends_densest10m_share")})
    P.table(pd.DataFrame(rows), "geometry", "Network geometry for simulation: carriageway width, measured stretch, "
            "edge drift, and access points (mid-road starts or ends bunched in one 10 m stretch locate side roads "
            "or exits).", main=False)

    rows = []
    for s in sites:
        trk = R[s]["trk"].assign(block=lambda x: (x["t0"] // 600).astype(int))
        for b, g in trk.groupby("block"):
            span = min(600.0, R[s]["tmax"] - b * 600)
            if span < 300:                                    # partial final block: too short for a demand rate
                continue
            row = {"Site": s, "Block (10 min)": int(b), "Total (veh/h)": int(round(len(g) / span * 3600))}
            for c in CLASSES:
                row[f"{c} (veh/h)"] = int(round((g["cls"] == c).sum() / span * 3600))
            rows.append(row)
    P.table(pd.DataFrame(rows), "demand", "Entering flow by class and 10-minute block (blocks shorter than 5 min "
            "omitted). In congestion (DEL_2) this is the discharge into the recorded stretch, not the upstream "
            "demand; the queue is set by a downstream bottleneck outside the view, so a simulation also needs a "
            "downstream boundary condition.", main=False)

    rows = []
    for s in sites:
        lf = R[s]["lf"]
        ok = lf["speed_kmh"] >= FREE_REGIME_MIN_KMH
        for c in CLASSES:
            x_all = lf.loc[ok & (lf["cls"] == c), "speed"]
            x = x_all[x_all >= STOPPED_KMH]
            if len(x) >= 50:
                rows.append({"Site": s, "Class": c, "Samples": len(x), "Stopped excluded (%)": round(100 * (1 - len(x) / len(x_all)), 1),
                             "Mean (km/h)": round(x.mean(), 1), "SD (km/h)": round(x.std(), 1),
                             "p15": round(x.quantile(0.15), 1), "p50": round(x.quantile(0.5), 1), "p85": round(x.quantile(0.85), 1),
                             "CV": round(x.std() / x.mean(), 2)})
    P.table(pd.DataFrame(rows), "free_speeds", f"Desired-speed targets: speeds of vehicles with no overlapping leader "
            f"within {LEADER_RADIUS_M:g} m, in windows with space-mean speed ≥ {FREE_REGIME_MIN_KMH:g} km/h, excluding "
            f"stopped or dwelling vehicles (< {STOPPED_KMH:g} km/h). A speed factor needs the posted limit, which the "
            "release does not state.", main=False)

    rows = []
    for s in sites:
        trk, sd = R[s]["trk"], R[s]["sd"]
        for c in CLASSES:
            x = trk.loc[trk["cls"] == c, "y_norm"]
            cl = sd.loc[(sd["cls"] == c) & (sd["v"] > 1), "clearance_m"]
            if len(x) >= 20:
                rows.append({"Site": s, "Class": c, "Tracks": len(x),
                             "Lateral p10 (m from edge 0)": round(x.quantile(0.1) * R[s]["width"], 2),
                             "Lateral p50 (m)": round(x.quantile(0.5) * R[s]["width"], 2),
                             "Lateral p90 (m)": round(x.quantile(0.9) * R[s]["width"], 2),
                             "Side clearance p05 (m)": round(q(cl, 0.05), 2), "Side clearance p50 (m)": round(q(cl, 0.5), 2),
                             "Side samples": len(cl)})
    P.table(pd.DataFrame(rows), "lateral_targets", "Lateral targets: track-median distance from edge 0 of the main "
            "carriageway (edge 0 = the edge with the lower lateral coordinate) and lateral clearance to the "
            "nearest side neighbor while moving (typical widths).", main=False)

    rows = []
    for s in sites:
        ld, acc = R[s]["ld"], R[s]["acc"]
        for c in CLASSES:
            l = ld[ld["f_cls"] == c]
            jam = l.loc[(l["v_f"] < 1 / 3.6) & (l["v_l"] < 1 / 3.6) & (l["gap_m"] > 0) & (l["gap_m"] < JAM_MAX_GAP_M)
                        & (l["regime"] == REGIMES[0][2]), "gap_m"]
            tg = l.loc[(l["v_f"] > 20 / 3.6) & (l["gap_m"] > 0), "time_gap_s"]
            a = acc.loc[acc["cls"] == c]
            start = a.loc[(a["speed"] < 15) & (a["lon_acc"] > 0), "lon_acc"]
            if len(l) >= 50:
                rows.append({"Site": s, "Class": c,
                             "Standstill pairs": len(jam),
                             "Standstill gap p50 (m)": round(q(jam, 0.5), 2) if len(jam) >= 30 else "—",
                             "Standstill gap p10 (m)": round(q(jam, 0.1), 2) if len(jam) >= 30 else "—",
                             "Time gap p50 (s)": round(q(tg, 0.5), 2), "Time gap p15 (s)": round(q(tg, 0.15), 2),
                             "Start-up accel p95 (m/s²)": round(q(start, 0.95), 2) if len(start) >= 50 else "—",
                             "Accel p99 (m/s²)": round(q(a["lon_acc"], 0.99), 2),
                             "Decel p01 (m/s²)": round(q(a["lon_acc"], 0.01), 2)})
    P.table(pd.DataFrame(rows), "longitudinal_targets", "Longitudinal targets: standstill gap (follower and leader "
            f"both stopped, gap < {JAM_MAX_GAP_M:g} m, congested windows only), time gap above 20 km/h, start-up "
            "acceleration (speed < 15 km/h), and tail percentiles of acceleration. The released positions are "
            "smoothed, so acceleration tails are lower bounds of vehicle capability, not maximum rates.", main=False)

    rows = []
    for s in sites:
        w = R[s]["win"].assign(block=lambda x: (x["win"] * WINDOW_S // 600).astype(int))
        for b, g in w.groupby("block"):
            rc = g["regime"].replace({REGIMES[2][2]: "uncongested ≥25", REGIMES[3][2]: "uncongested ≥25"}).value_counts(normalize=True)
            rows.append({"Site": s, "Block (10 min)": int(b), "Windows": len(g),
                         "Flow (veh/h)": int(round(g["flow_vph"].mean())), "Density (veh/km)": round(g["density_vpkm"].mean(), 1),
                         "Speed (km/h)": round(g["speed_kmh"].mean(), 1),
                         "Dominant regime": rc.index[0], "Share of dominant regime (%)": round(100 * rc.iloc[0]),
                         "Use": "fit/validate" if (len(g) >= 10 and rc.iloc[0] >= 0.8) else "exclude (mixed or short)"})
    P.table(pd.DataFrame(rows), "validation_targets", "Traffic-state targets per 10-minute block (moderate and free "
            "windows pooled as uncongested). Blocks that mix congested, slow and uncongested windows or have fewer "
            "than 10 windows should not be used as single targets; split fitting and validation blocks within each regime.", main=False)

    fig, axs = plt.subplots(len(sites), 1, figsize=(W1, 1.9 * len(sites) + 0.4), squeeze=False)
    for a, s in zip(axs[:, 0], sites):
        o = R[s]["occ"]
        a.hexbin(o["s"], o["yl"], gridsize=(80, 14), bins="log", cmap="Blues", mincnt=1, linewidths=0)
        a.set_title(s, loc="left"); a.set_ylabel("lateral (m)")
    axs[-1, 0].set_xlabel("distance along road (m)")
    P.figure(fig, "occupancy_maps", "Occupancy of the main carriageway (darker = more rows, logarithmic). Isolated "
             "dark cells at the edges are vehicles standing at the curb.", main=False)

    fig, axs = plt.subplots(1, len(sites), figsize=(W1, 3.0), sharey=True, squeeze=False)
    for a, s in zip(axs[0], sites):
        lf = R[s]["lf"]
        f = lf[(lf["speed_kmh"] >= FREE_REGIME_MIN_KMH) & (lf["speed"] >= STOPPED_KMH)]
        for c in ["Car", "M2W", "M3W"]:
            x = np.sort(f.loc[f["cls"] == c, "speed"].to_numpy())
            if len(x) > 50:
                a.plot(x, np.arange(1, len(x) + 1) / len(x), color=CLS_COLOR[c], label=c)
        a.set_title(s, loc="left"); a.set_xlabel("desired speed (km/h)")
    axs[0][0].set_ylabel("cumulative share"); axs[0][0].legend(fontsize=7)
    P.figure(fig, "free_speed_ecdf", "Desired-speed distributions by class (leader-free vehicles in uncongested "
             f"windows, stopped vehicles excluded).", main=False)

    fig, axs = plt.subplots(1, len(sites), figsize=(W1, 3.0), sharey=True, squeeze=False)
    for a, s in zip(axs[0], sites):
        trk = R[s]["trk"]
        for c in ["Car", "M2W", "M3W"]:
            x = trk.loc[trk["cls"] == c, "y_norm"].clip(0, 1)
            if len(x) > 20:
                a.hist(x, bins=20, density=True, histtype="step", lw=1.8, color=CLS_COLOR[c], label=c)
        a.set_title(s, loc="left"); a.set_xlabel("lateral position (0–1)")
    axs[0][0].set_ylabel("probability density"); axs[0][0].legend(fontsize=7)
    P.figure(fig, "lateral_by_class", f"Where each class travels across the main carriageway (track medians; "
             f"{lateral_label(sites)}).", main=False)

    s = "DEL_2" if "DEL_2" in sites else sites[0]
    ld = R[s]["ld"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 3.0))
    over = []
    for c in ["Car", "M2W", "M3W"]:
        l = ld[ld["f_cls"] == c]
        jam = l.loc[(l["v_f"] < 1 / 3.6) & (l["v_l"] < 1 / 3.6) & (l["gap_m"] > 0) & (l["gap_m"] < JAM_MAX_GAP_M)
                    & (l["regime"] == REGIMES[0][2]), "gap_m"]
        tg = l.loc[(l["v_f"] > 20 / 3.6) & (l["gap_m"] > 0), "time_gap_s"]
        if len(jam) > 20:
            a1.hist(jam, bins=np.arange(0, 5.01, 0.2), density=True, histtype="step", lw=1.8, color=CLS_COLOR[c], label=c)
            over.append(f"{c} {100 * (jam > 5).mean():.0f}%")
        if len(tg) > 20:
            a2.hist(tg[tg <= 6], bins=30, density=True, histtype="step", lw=1.8, color=CLS_COLOR[c], label=c)
    a1.set_xlabel("standstill gap (m)"); a1.set_ylabel("probability density"); a1.set_title(f"(a) standstill gap, {s}", loc="left")
    a1.legend(fontsize=7); a2.set_xlabel("time gap (s)"); a2.set_title(f"(b) time gap above 20 km/h, {s}", loc="left")
    P.figure(fig, "gap_distributions", "Car-following targets by follower class: (a) standstill gaps in congested "
             f"windows, shown to 5 m (beyond 5 m: {', '.join(over) or 'none'}); (b) time gaps up to 6 s.", main=False)

    fig, axs = plt.subplots(1, len(sites), figsize=(W1, 3.0), sharey=False, squeeze=False)
    for a, s in zip(axs[0], sites):
        w = R[s]["win"].assign(block=lambda x: (x["win"] * WINDOW_S // 600).astype(int))
        for o in [r[2] for r in REGIMES]:
            ww = w[w["regime"] == o]
            if len(ww):
                a.scatter(ww["density_vpkm"], ww["flow_vph"], s=7, color=REGIME_COL[o], label=f"{o} km/h")
        a.set_title(s, loc="left"); a.set_xlabel("density (veh/km)")
    axs[0][0].set_ylabel("flow (veh/h)")
    hl = {}
    for a in axs[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            hl.setdefault(ll, hh)
    fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=4, fontsize=7)
    P.figure(fig, "fd_targets", "Flow–density targets: 30 s windows colored by regime.", main=False)


# %% 7 · Run — one combined paper; main-text exhibits in <key>_main.pdf, everything in <key>.pdf
SUPP_ORDER = {
    "F": ["trim_sensitivity", "acc_raw_vs_clean", "lateral_corridors_noida", "footprints", "fd_per_metre",
          "table2_label_check", "mid_road_ends", "speed_density_by_class", "lateral_by_class", "occupancy_maps",
          "free_speed_ecdf", "gap_distributions", "fd_targets"],
    "T": ["documentation_vs_data", "cleaning_log", "footprints", "trim_sensitivity", "speed_before_after",
          "corridor_gap_sensitivity", "sample", "speed_by_class", "passing_detail", "clearance_close_passes",
          "width_sensitivity", "dynamic_pcu", "geometry", "demand", "free_speeds", "lateral_targets",
          "longitudinal_targets", "validation_targets"],
}
if "REPLACE" in TYPICAL_DIMS_SOURCE:
    print("WARNING: TYPICAL_DIMS are placeholders — set them from a citable source and fill TYPICAL_DIMS_SOURCE.")
if not all(MEDIAN_EDGE.get(s) in ("low", "high") for s in DELHI):
    print("WARNING: MEDIAN_EDGE not confirmed for every Delhi site — lateral results are labelled 'orientation not confirmed'.")
PACK = Pack(PAPER_KEY, PAPER_TITLE,
            "Part 1: audit of the released files, cleaning workflow, and the cost of skipping it (all six sites). "
            "Part 2: two-wheeler filtering on the cleaned main carriageways of DEL_2 (congested) and DEL_3 "
            "(uncongested reference). Supplement: audit details, filtering details, and calibration-ready targets "
            "for simulation. Typical vehicle sizes (" + TYPICAL_DIMS_SOURCE + "): " +
            ", ".join(f"{k} {v[0]}×{v[1]} m" for k, v in TYPICAL_DIMS.items() if k in CLASSES) + ".",
            split=True, supp_order=SUPP_ORDER)
for k, fn in [("audit", pack_p1), ("filtering", pack_p4), ("targets", pack_p2)]:
    if k in RUN_PARTS:
        print(f"\n=== {k} ===")
        fn(PACK)
        gc.collect()
PACK.close()
print("\nPaper pack in", PACK.dir)
