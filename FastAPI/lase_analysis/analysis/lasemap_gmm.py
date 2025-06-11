
# ---------------------------------------------------------------------------
#  lasemap.py  –  Laser-particle map analysis (fully patched 2025-04-24)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from copy import deepcopy
from time import time
from os import cpu_count
from joblib import Parallel, delayed

from sklearn.neighbors import radius_neighbors_graph
from sklearn.mixture import GaussianMixture
from scipy.sparse.csgraph import connected_components

from .mapline             import MapLine, LineSplitter          # noqa: F401  (kept for API parity)
from ..data               import LaseData_Map_Confocal
from ..data.spectrum      import Spectrum, PeakFitOpts
from ..utils.logging       import logger
from ..utils.constants     import LIDS, MIDS, COLS, SDTP, K_nm2meV as KE

# ---------------------------------------------------------------------------
# helper utilities – safe for uint64 and robust against NumPy ufunc bugs
# ---------------------------------------------------------------------------

def _mask_isin(series: pd.Series, values) -> np.ndarray:
    """
    Boolean mask equivalent to `series.isin(values)` but avoids the NumPy
    '&' / '|' casting bug that appears with unsigned-int dtypes.
    """
    if series.dtype == np.uint64:
        series = series.astype(np.int64)
        values = np.asarray(values, dtype=np.int64)
    return np.isin(series.values, values, assume_unique=False)

def _reshape_row(arr):
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

# ---------------------------------------------------------------------------
#  dual-mode recognition
# ---------------------------------------------------------------------------

def _is_dual_mode(energies: np.ndarray,
                  gap_lo: float = 40.0,
                  gap_hi: float = 80.0) -> bool:
    """
    True if `energies` contains exactly two peaks whose absolute difference
    falls in the [gap_lo, gap_hi] meV range (default: 40-80 meV).

    The bounds are intentionally generous because the cavity FSR varies
    a few meV with disk size and local index.  Adjust in find_lines_and_multi
    via the `dual_gap` parameter if needed.
    """
    if len(energies) != 2:
        return False
    gap = abs(float(energies[1]) - float(energies[0]))
    return gap_lo <= gap <= gap_hi

# ---------------------------------------------------------------------------
#  Parameter classes
# ---------------------------------------------------------------------------

class PeaksOpts:
    """Options for Spectrum.fitting + local code."""
    OPTS = {"pool": False, "chunk": 100_000, "method": 2}

    def __init__(self, **kw):
        for k, v in self.OPTS.items():
            setattr(self, k, kw.get(k, v))
        # cascade Spectrum-fit options
        for k, v in PeakFitOpts.OPTS.items():
            setattr(self, k, kw.get(k, v))

    @property
    def fit_options(self):
        return PeakFitOpts(**{k: getattr(self, k) for k in PeakFitOpts.OPTS})

# ---------------------------------------------------------------------------
#  Main analysis class
# ---------------------------------------------------------------------------

class LaseAnalysis_Map:
    """Peak → Line → Multiplet analysis for confocal map data."""
    FWHME0 = 0.615  # Lorentzian width used for line weighting

    # ------------------- constructor / storage -----------------------------
    def __init__(self, lfile, gname, pks=None, lns=None, mlt=None):
        self.lfile = lfile
        self.name  = gname
        self.pks   = pks
        self.lns   = lns
        self.mlt   = mlt
        self.info  = deepcopy(lfile.info[self.name])

    # ------------------- public IO helpers --------------------------------
    def save_analysis(self, analysis="base", overwrite=False):
        self.lfile.save_analysis(self, analysis, overwrite)

    def get_data(self, peaks=True):
        if self.lns is None or self.mlt is None:
            raise RuntimeError("run find_lines_and_multi() first")
        pks = self.pks[self.pks.lid.isin(self.lns.index)] if peaks else None
        return LaseData_Map_Confocal(self.mlt, self.lns, pks, name=self.name)

    # ----------------------------------------------------------------------
    #  1.  peak finding
    # ----------------------------------------------------------------------
    def find_peaks(self, popt=None, **kw):
        logger.info("────────── Find peaks ──────────")
        t0   = time()
        popt = popt or PeaksOpts(**kw)

        with self.lfile.read() as f:
            wl_axis = f["data"]["wl_axis"][:].astype(np.float64)
            coords  = f["data"]["coordinates"][:, 4]

        idxs = np.where(coords == self.info.id)[0]
        chunks = [idxs] if popt.chunk is None else [
            idxs[i:i+popt.chunk] for i in range(0, len(idxs), popt.chunk)
        ]

        # choose spectrum-analysis function
        if popt.method == 1:
            analyzer = self._analyze_spt
            opts = [popt.med_filter, popt.threshold, popt.window,
                    self.FWHME0, popt.gain, popt.saturation]
        else:
            step = wl_axis[1] - wl_axis[0]
            clst = int(1.5 * popt.window / step)
            analyzer = self._analyze_spt2
            opts = [popt.med_filter, popt.threshold, popt.prominence,
                    popt.distance, clst, popt.saturation,
                    popt.window, self.FWHME0, popt.gain]

        pool   = Parallel(cpu_count()-2) if popt.pool else None
        frames = []

        for sl in chunks:
            with self.lfile.read() as f:
                spectra = f["data"]["spectra"][sl].astype(np.float64)

            if pool:
                out = pool(delayed(analyzer)(wl_axis, y, sl[i], *opts)
                           for i, y in enumerate(spectra))
            else:
                out = [analyzer(wl_axis, y, sl[i], *opts)
                       for i, y in enumerate(spectra)]
            rows = [_reshape_row(r) for r in out if r is not None and len(r)]
            if rows:
                frames.append(pd.DataFrame(np.vstack(rows), columns=COLS.FIT))

        if not frames:
            self.pks = pd.DataFrame(columns=list(COLS.MPKS))
            logger.info("No peaks found [t=%.3fs]", time()-t0)
            return

        df = (pd.concat(frames, ignore_index=True)
                .dropna()
                .reset_index(drop=True)
                .astype({"a":np.float32, "wl":np.float32,
                         "fwhm":np.float32, "ph":np.float32, "ispt":int}))

        # ---------- strict de-duplication (round-hash) ---------------------
        wl_rnd = (df.wl * 100).round().astype(int)   # 0.01 nm bins
        hsh    = (df.ispt.values.astype(np.uint64) << 32) + wl_rnd.values
        df     = df.loc[~pd.Series(hsh).duplicated()].reset_index(drop=True)

        # ---------- spatial coords, enrich -------------------------------
        with self.lfile.read() as f:
            xyz_all = f["data"]["coordinates"][:,:3]
        xyz = xyz_all[df.ispt.values]

        pks = pd.concat([pd.DataFrame(xyz, columns=["i","j","k"]), df], axis=1)
        pks["lid"]   = LIDS.NON
        pks["E"]     = (KE / pks.wl).astype(np.float32)
        pks["fwhmE"] = (pks.fwhm * pks.E / pks.wl).astype(np.float32)

        pks = (pks.replace([np.inf,-np.inf], np.nan)
                   .dropna(subset=["i","j","k"])
                   .reset_index(drop=True))

        pks.sort_values(["ispt","E"], inplace=True)
        pks.index = pd.Index(range(len(pks)), name="pid", dtype=np.uint64)
        self.pks  = pks.astype(COLS.MPKS)

        logger.info("Peaks: %d  [t=%.3fs]", len(self.pks), time()-t0)

    # -- static wrappers around spectrum routines --------------------------
    @staticmethod
    def _analyze_spt(x, y, ispt, medflt, thr, wdw, fwhm0, gain, sat):
        y,_ = Spectrum.spt_filter(y, medflt)
        m   = Spectrum.spt_maxima(y, thr)
        return Spectrum.spt_fit(x, y, ispt, m, wdw, fwhm0, gain, sat)

    @staticmethod
    def _analyze_spt2(x, y, ispt, medflt, thr, prm, dst, clst, sat, wdw, fwhm0, gain):
        y,_ = Spectrum.spt_filter(y, medflt)
        m   = Spectrum.spt_maxima2(y, thr, prm, dst, clst, sat)
        return Spectrum.spt_fit2(x, y, ispt, m, wdw, fwhm0, gain, sat)

    # April 25, simplified over-fitted version
    # ----------------------------------------------------------------------
    #  2.  build Lines + Multiplets with dual-mode handling
    # ----------------------------------------------------------------------
    # def find_lines_and_multi(self,
    #                          spatial_radius: float = 3.0,
    #                          merge_radius:   float = 12.0,
    #                          bic_max_comp:   int   = 8,
    #                          dual_gap: tuple = (40.0, 80.0)):
    #     """
    #     *spatial_radius* : neighbour radius (pixels) for CC graph  
    #     *merge_radius*   : radius to merge fragmented lines into a mid  
    #     *bic_max_comp*   : max Gaussian components to try per CC  
    #     *dual_gap*       : (low, high) meV window for dual-mode detection
    #     """
    #     if self.pks is None or self.pks.empty:
    #         raise RuntimeError("run find_peaks first")

    #     logger.info("────────── Cluster peaks ──────────")
    #     t0 = time()

    #     # ---- 1. spatial CC -------------------------------------------------
    #     pks = self.pks.copy()
    #     xyz = pks[["i","j","k"]].to_numpy(float)
    #     G   = radius_neighbors_graph(xyz, radius=spatial_radius,
    #                                  include_self=True, n_jobs=-1)
    #     _, ccid = connected_components(G, directed=False)
    #     pks["ccid"] = ccid.astype(np.uint32)

    #     # ---- 2a. collapse dual-mode events --------------------------------
    #     gap_lo, gap_hi = dual_gap
    #     for cid, grp in pks.groupby("ccid"):
    #         e_sorted = np.sort(grp.E.values)
    #         if _is_dual_mode(e_sorted, gap_lo, gap_hi):
    #             # keep the stronger peak, drop the weaker
    #             idx_keep = grp.index[np.argmax(grp.a.values)]
    #             idx_drop = grp.index.difference([idx_keep])
    #             pks.loc[idx_drop, "drop_me"] = True
    #     if "drop_me" in pks:
    #         pks = pks.loc[pks.drop_me.isna()].drop(columns="drop_me")

    #     # ---- 2b. per-CC Gaussian mixture → lid ----------------------------
    #     lids = np.empty(len(pks), dtype=np.uint64)
    #     next_id = 0
    #     for cid, grp in pks.groupby("ccid"):
    #         Es = grp.E.values.reshape(-1,1)
    #         if len(Es) == 1 or np.nanstd(Es) < 1e-6:
    #             lids[grp.index] = next_id
    #             next_id += 1
    #             continue
    #         best_lbl, best_bic = None, np.inf
    #         max_comp = min(bic_max_comp, len(Es))
    #         for k in range(1, max_comp+1):
    #             if len(Es) < 2*k:
    #                 continue
    #             gm = GaussianMixture(k, covariance_type="diag",
    #                                  reg_covar=1e-3, random_state=0)
    #             gm.fit(Es)
    #             bic = gm.bic(Es)
    #             if bic < best_bic:
    #                 best_bic, best_lbl = bic, gm.predict(Es)
    #         lids[grp.index] = best_lbl + next_id
    #         next_id += best_lbl.max() + 1
    #     pks["lid"] = lids

    #     # ---- 3. build Line table ------------------------------------------
    #     groups = pks.groupby("lid")
    #     lns_arr = [self._lines_info(g, self.FWHME0) for _, g in groups]
    #     lns = pd.DataFrame(np.concatenate(lns_arr),
    #                        index=list(groups.groups),
    #                        columns=[n for n,_ in SDTP.MLNS]).astype(COLS.MLNS)
    #     lns.index.name = "lid"

    #     # ---- 4. merge nearby lines → mid ----------------------------------
    #     centres = lns[["i","j","k"]].to_numpy()
    #     if len(centres) < 2:
    #         mids = np.zeros(len(centres), dtype=np.uint64)
    #     else:
    #         Gm = radius_neighbors_graph(centres, radius=merge_radius,
    #                                     include_self=False, n_jobs=-1)
    #         _, mids = connected_components(Gm, directed=False)
    #     lns["mid"] = mids.astype(np.uint64)

    #     # propagate mid to peaks
    #     pks["mid"] = pks.lid.map(lns.mid).astype(np.uint64)

    #     # ---- 5. Multiplets -------------------------------------------------
    #     m_arr = [self._multi_info(sub) for _, sub in lns.groupby("mid")]
    #     mlt = pd.DataFrame(np.concatenate(m_arr),
    #                        index=lns.mid.unique(),
    #                        columns=[n for n,_ in SDTP.MMLT]).astype(COLS.MMLT)
    #     mlt.index.name = "mid"

    #     # ---- 6. store ------------------------------------------------------
    #     self.lns, self.mlt, self.pks = lns, mlt, pks
    #     logger.info("Lines: %d  Multiplets: %d  [t=%.3fs]",
    #                 len(lns), len(mlt), time()-t0)


    # april 29 detailed version
    # ----------------------------------------------------------------------
    #  2.  build Lines + Multiplets  (GMM + all DBSCAN safeguards)
    # ----------------------------------------------------------------------
    def find_lines_and_multi(
            self,
            spatial_radius : float = 3.0,      # px – CC graph in 4-D
            energy_scale   : float = 0.5,      # 1 px  ≃ 0.5 meV  (from old lopt.scale)
            merge_radius   : float = 12.0,     # px – spatial merge of line centroids
            bic_max_comp   : int   = 8,        # max components tried in each GMM
            min_pts        : int   = 2,        # ≥ peaks per GMM component / line
            var_floor_meV  : float = 0.25,     # σ_E floor → avoids σ→0 splitting
            dual_gap       : tuple = (40.,80.),# keep doublets, don’t drop
            min_overlap    : float = .33       # same as old mopt.min_overlap
        ):
        """
        Re-implementation of the old DBSCAN logic with a GMM core.

        – spatial CC built in **4-D (i,j,k,E′)** where E′ = E·energy_scale  
        – each component from the GMM is kept **only if it owns ≥ min_pts peaks**  
          else it is merged into its nearest-mean neighbour.  
        – a variance floor `var_floor_meV` (meV) stops GMM from inventing
          near-delta components on the wings of broad lines.  
        – MIDs are the connected components of a graph whose edges require  
            distance(i,j,k) < merge_radius  **and** peaks-overlap ≥ min_overlap
        """
        if self.pks is None or self.pks.empty:
            raise RuntimeError("run find_peaks first")

        t0 = time();  logger.info("────────── Cluster peaks (GMM+safe) ──────────")

        # ──────────────────────────────────────────────────────────────────
        # 1) preliminary filtering  (reuse thresholds you passed in lopt)
        # ------------------------------------------------------------------
        pks = self.pks.copy()
        lopts = getattr(self, "_line_filter_opts", None)   # set by user if wanted
        if lopts is not None:
            bad = np.zeros(len(pks), dtype=bool)
            for key,(lo,hi) in lopts.items():
                bad |= (pks[key] < lo) | (pks[key] > hi)
            pks.loc[bad,"lid"] = LIDS.FLT
            pks = pks.loc[~bad]

        if pks.empty:
            self.lns = self._empty_lns(); self.mlt = self._empty_mlt(); return

        # ──────────────────────────────────────────────────────────────────
        # 2) build 4-D neighbour graph  → CC ids
        # ------------------------------------------------------------------
        xyzE = pks[["i","j","k","E"]].to_numpy(float)
        xyzE[:,3] *= energy_scale                     # scale energy
        G = radius_neighbors_graph(xyzE, radius=spatial_radius,
                                  include_self=True, n_jobs=-1)
        _, ccid = connected_components(G, directed=False)
        pks["ccid"] = ccid.astype(np.uint32)

        # ──────────────────────────────────────────────────────────────────
        # 3) dual-mode *tag* (do NOT drop any peak!)
        # ------------------------------------------------------------------
        gap_lo,gap_hi = dual_gap
        pks["is_dual"] = False
        for cid,grp in pks.groupby("ccid"):
            if len(grp)==2 and gap_lo <= abs(grp.E.iloc[1]-grp.E.iloc[0]) <= gap_hi:
                pks.loc[grp.index,"is_dual"] = True    # just diagnostic

        # ──────────────────────────────────────────────────────────────────
        # 4) per-CC GMM with safeguards  → initial component labels
        # ------------------------------------------------------------------
        lids   = np.empty(len(pks), dtype=np.uint64)
        nextID = 0
        var_floor = (var_floor_meV**2)                 # meV²
        for cid,grp in pks.groupby("ccid"):
            Es = grp.E.values.reshape(-1,1)
            n  = len(Es)
            if n==1 or np.nanstd(Es) < 1e-6:
                lids[grp.index] = nextID; nextID += 1; continue

            best_lbl,best_bic = None,np.inf
            for k in range(1, min(bic_max_comp,n)+1):
                if n < 2*k: break
                gm = GaussianMixture(
                        k, covariance_type="diag",
                        reg_covar=var_floor,            # ← variance floor
                        random_state=0)
                gm.fit(Es)
                bic = gm.bic(Es)
                if bic < best_bic:
                    best_bic = bic; best_lbl = gm.predict(Es)

            # ── enforce min_pts per component
            # lbl = best_lbl.copy()
            # for lab,cnt in zip(*np.unique(lbl,return_counts=True)):
            #     if cnt < min_pts:
            #         # merge into nearest-mean component
            #         my_mu    = Es[lbl==lab].mean()
            #         others   = {l:abs(my_mu-Es[lbl==l].mean()) for l in np.unique(lbl) if l!=lab}
            #         target   = min(others,key=others.get)
            #         lbl[lbl==lab] = target
            # # relabel to 0..m-1 then offset
            # u = np.unique(lbl); m = {l:i for i,l in enumerate(u)}
            # final = np.array([m[x] for x in lbl],dtype=int)
            # lids[grp.index] = final + nextID
            # nextID += final.max()+1
            # ── enforce min_pts per component ────────────────────────────────
            # lbl   = best_lbl.copy()
            # lbl_u = np.unique(lbl)

            # # Only attempt to merge tiny components if there are ≥2 labels
            # if lbl_u.size > 1:
            #     # get counts for every possible label
            #     counts = np.bincount(lbl, minlength=lbl_u.max()+1)
            #     for lab in lbl_u:
            #         if counts[lab] < min_pts:
            #             # find all other labels
            #             others = [l for l in lbl_u if l != lab]
            #             if not others:
            #                 # no other component to merge into—skip
            #                 continue
            #             # pick the other component whose mean E is closest
            #             my_mu  = Es[lbl == lab].mean()
            #             target = min(others,
            #                         key=lambda l: abs(my_mu - Es[lbl == l].mean()))
            #             lbl[lbl == lab] = target

            # # now relabel to 0..m-1 and offset as before
            # u      = np.unique(lbl)
            # mapping = {old: new for new, old in enumerate(u)}
            # final  = np.array([mapping[x] for x in lbl], dtype=int)
            # lids[grp.index] = final + nextID
            # nextID += final.max() + 1

            # enforce min_pts per component
            lbl    = best_lbl.copy()
            lbl_u  = np.unique(lbl)

            # only if there’s more than one component
            if lbl_u.size > 1:
                # pre-compute counts so we only try real labels
                counts = {l: int(np.sum(lbl==l)) for l in lbl_u}

                for lab, cnt in counts.items():
                    if cnt < min_pts:
                        # pick only those l that still have points
                        valid_others = [l for l, c in counts.items() if l != lab and c > 0]
                        if not valid_others:
                            # nothing to merge into — skip
                            continue

                        my_mu = Es[lbl == lab].mean()
                        # now compute each other’s mean safely
                        dist_to_other = {
                            l: abs(my_mu - Es[lbl == l].mean())
                            for l in valid_others
                        }
                        target = min(dist_to_other, key=dist_to_other.get)
                        lbl[lbl == lab] = target

            # relabel to 0..m-1 then offset
            u       = np.unique(lbl)
            mapping = {old: new for new, old in enumerate(u)}
            final   = np.array([mapping[x] for x in lbl], dtype=int)
            lids[grp.index] = final + nextID
            nextID       += final.max() + 1



        pks["lid"] = lids

        # drop any line with <min_pts peaks (safety net)
        keep = pks.groupby("lid").size()      # number of peaks in each lid
        keep = keep[keep >= min_pts].index    # lids that pass the threshold
        pks  = pks[pks.lid.isin(keep)]


        # keep = pks.groupby("lid").pid.count() >= min_pts
        # pks = pks[pks.lid.isin(keep[keep].index)]

        # ──────────────────────────────────────────────────────────────────
        # 5) build Line table  (same weighted logic as before)
        # ------------------------------------------------------------------
        groups = pks.groupby("lid")
        larr   = [self._lines_info(g,self.FWHME0) for _,g in groups]
        lns = pd.DataFrame(np.concatenate(larr),
                          index=list(groups.groups),
                          columns=[n for n,_ in SDTP.MLNS]).astype(COLS.MLNS)
        lns.index.name = "lid"

        # ──────────────────────────────────────────────────────────────────
        # 6) MID merging  = spatial radius **AND** intensity-overlap
        # ------------------------------------------------------------------
        centres = lns[["i","j","k"]].to_numpy()
        if len(centres)<2:
            mids = np.zeros(len(centres),dtype=np.uint64)
        else:
            Gm = radius_neighbors_graph(centres, radius=merge_radius,
                                        include_self=False, n_jobs=-1)
            # keep only edges whose peak-overlap ≥ min_overlap
            lids_idx = lns.index.to_numpy()
            for r,c in zip(*Gm.nonzero()):
                p0,p1 = lids_idx[r], lids_idx[c]
                pk0,pk1 = pks[pks.lid==p0][["ispt","a"]], pks[pks.lid==p1][["ispt","a"]]
                a0,a1   = pk0.a/ pk0.a.max(), pk1.a/ pk1.a.max()
                _,i0,i1 = np.intersect1d(pk0.ispt, pk1.ispt, return_indices=True)
                ovl     = 0 if len(i0)==0 else np.minimum(a0.values[i0],a1.values[i1]).sum() / min(a0.sum(),a1.sum())
                if ovl < min_overlap: Gm[r,c] = 0
            _,mids = connected_components(Gm, directed=False)
        lns["mid"] = mids.astype(np.uint64)
        pks["mid"] = pks.lid.map(lns.mid).astype(np.uint64)

        # guard: nukes singleton MIDs that have only one line AND that line has <min_pts peaks
        # good_mid = lns.groupby("mid").lid.count() > 1

        counts   = lns.groupby("mid").size()        # number of lines per mid
        good_mid = counts.index #counts[counts > 1].index         # mids with ≥2 lines
        # bad_mid  = good_mid[~good_mid].index
      # filter out the bad mids from both lns and pks
        lns = lns[lns.mid.isin(good_mid)]
        pks = pks[pks.mid.isin(good_mid)]

        # pks = pks[~pks.mid.isin(bad_mid)]
        # lns = lns[~lns.mid.isin(bad_mid)]

        # ──────────────────────────────────────────────────────────────────
        # 7) build Multiplet table
        # ------------------------------------------------------------------
        m_arr = [self._multi_info(s) for _,s in lns.groupby("mid")]
        mlt = pd.DataFrame(np.concatenate(m_arr),
                          index=lns.mid.unique(),
                          columns=[n for n,_ in SDTP.MMLT]).astype(COLS.MMLT)
        mlt.index.name = "mid"

        # save
        self.pks, self.lns, self.mlt = pks, lns, mlt
        logger.info("Lines:%d  Multiplets:%d  (%.1f s)",
                    len(lns), len(mlt), time()-t0)


    # ----------------------------------------------------------------------
    # static helpers for line / multiplet info
    # ----------------------------------------------------------------------
    @staticmethod
    def _lines_info(pks, fwhmE0):
        E0 = np.average(pks.E, weights=1/np.power(0.1 +
                        np.abs(pks.fwhmE-fwhmE0), 2))
        return np.array([(
            np.average(pks.i, weights=pks.a).astype(np.float32),
            np.average(pks.j, weights=pks.a).astype(np.float32),
            np.average(pks.k, weights=pks.a).astype(np.float32),
            pks.a.sum().astype(np.float32),
            (KE/E0).astype(np.float32),
            np.std(pks.wl.values).astype(np.float32),
            E0.astype(np.float32),
            np.std(pks.E.values).astype(np.float32),
            pks.ph.sum().astype(np.float32),
            pks.shape[0],
            (pks.i.max()-pks.i.min() + pks.j.max()-pks.j.min())
               .astype(np.float32),
            MIDS.NON
        )], dtype=SDTP.MLNS)

    @staticmethod
    def _multi_info(lns):
        return np.array([(
            np.float32(lns.i.mean()),
            np.float32(lns.j.mean()),
            np.float32(lns.k.mean()),
            len(lns.index)
        )], dtype=SDTP.MMLT)

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------





# # import numpy as np
# # import pandas as pd

# # from scipy.spatial.distance import squareform
# # from sklearn.neighbors import radius_neighbors_graph
# # from sklearn.mixture import GaussianMixture
# # from scipy.sparse.csgraph import connected_components

# # from joblib import Parallel, delayed
# # from os import cpu_count
# # from itertools import combinations
# # from time import time
# # from copy import deepcopy

# # from .mapline import MapLine, LineSplitter
# # from ..data import LaseData_Map_Confocal
# # from ..data.spectrum import Spectrum, PeakFitOpts
# # from ..utils.logging import logger
# # from ..utils.constants import LIDS, COLS, SDTP, K_nm2meV as KE


# # class LaseAnalysis_Map:
# #     FWHME0 = 0.615

# #     def __init__(self, lfile, gname, pks=None, lns=None, mlt=None):
# #         self.lfile = lfile
# #         self.name = gname
# #         self.pks = pks
# #         self.lns = lns
# #         self.mlt = mlt
# #         self.info = deepcopy(lfile.info[self.name])

# #     @property
# #     def peaks(self):
# #         return self.pks

# #     @property
# #     def lines(self):
# #         return self.lns

# #     @property
# #     def multi(self):
# #         return self.mlt

# #     def save_analysis(self, analysis='base', overwrite=False):
# #         self.lfile.save_analysis(self, analysis, overwrite)

# #     def get_data(self, peaks=True):
# #         if self.lns is None or self.mlt is None:
# #             raise RuntimeError('Analysis missing lines or multi data')
# #         pks = self.pks[self.pks.lid.isin(self.lns.index)] if peaks else None
# #         return LaseData_Map_Confocal(multi=self.mlt, lines=self.lns, peaks=pks,
# #                                      name=self.name, feats=None)

# #     # ------------------------------ Peaks analysis ------------------------------

# #     def find_peaks(self, popt=None, **kwds):
# #         logger.info('---------- Find peaks ----------')
# #         t0 = time()
# #         if popt is None:
# #             popt = PeaksOpts(**kwds)

# #         ginfo = self.lfile.info[self.name]
# #         with self.lfile.read() as f:
# #             nspectra, npixels = f['data']['spectra'].shape
# #             x = f['data']['wl_axis'][:].astype(np.float64)
# #             gidxs = f['data']['coordinates'][:,4]

# #         cidx = np.where(gidxs == ginfo.id)[0]
# #         nn = len(cidx)
# #         idx_list = ([cidx] if popt.chunk is None else
# #                     [cidx[ii*popt.chunk : min((ii+1)*popt.chunk, nn)]
# #                      for ii in range(int(np.ceil(nn/popt.chunk)))])

# #         ffit = (LaseAnalysis_Map._analyze_spt if popt.method == 1
# #                 else LaseAnalysis_Map._analyze_spt2)
# #         clst = int(1.5 * popt.window / (x[npixels//2] - x[npixels//2-1]))
# #         if popt.method == 1:
# #             fopt = [popt.med_filter, popt.threshold, popt.window,
# #                     self.FWHME0, popt.gain, popt.saturation]
# #         else:
# #             fopt = [popt.med_filter, popt.threshold, popt.prominence,
# #                     popt.distance, clst, popt.saturation,
# #                     popt.window, self.FWHME0, popt.gain]
# #         pool = Parallel(cpu_count()-2) if popt.pool else None

# #         fit = []
# #         for ic, idx in enumerate(idx_list):
# #             with self.lfile.read() as f:
# #                 spectra = f['data']['spectra'][idx,:].astype(np.float64)
# #             if pool:
# #                 ret = pool(delayed(ffit)(x, y, idx[ii], *fopt)
# #                            for ii, y in enumerate(spectra))
# #             else:
# #                 ret = [ffit(x, y, idx[ii], *fopt)
# #                        for ii, y in enumerate(spectra)]
# #             if ret:
# #                 fit.append(pd.DataFrame(np.concatenate(ret), columns=COLS.FIT)
# #                           .dropna(how='any')
# #                           .astype({'ispt': int}))

# #         if not fit:
# #             self.pks = None
# #         else:
# #             fit = (pd.concat(fit, ignore_index=True)
# #                    .astype({'a': np.float32, 'wl': np.float32,
# #                             'fwhm': np.float32, 'ph': np.float32}))
# #             hsh = self._pks_hash(fit)
# #             uhs, cnt = np.unique(hsh, return_counts=True)
# #             if np.any(cnt > 1):
# #                 dup = [i for uu in uhs[cnt>1]
# #                            for i in np.where(hsh==uu)[0][1:]]
# #                 fit.drop(index=dup, inplace=True)
# #                 fit.reset_index(drop=True, inplace=True)
# #             crd = self._merged_coordinates(fit.ispt.values)
# #             lid = pd.Series(LIDS.NON, index=fit.index,
# #                             name='lid', dtype=np.uint64)
# #             pks = pd.concat([crd, fit, lid], axis=1)
# #             pks['E'] = (KE / pks['wl']).astype(np.float32)
# #             pks['fwhmE'] = (pks['fwhm'] * pks['E'] / pks['wl']).astype(np.float32)
# #             pks.sort_values(['ispt','E'], inplace=True)
# #             pks.reset_index(drop=True, inplace=True)
# #             pks.index.name = 'pid'
# #             self.pks = pks[list(COLS.MPKS)].astype(COLS.MPKS)

# #         logger.info(f'Done! [t={time()-t0:.3f}s]')

# #     @staticmethod
# #     def _analyze_spt(x, y, ispt, medflt, thr, wdw, fwhm0, gain, sat):
# #         y, _ = Spectrum.spt_filter(y, medflt)
# #         midx = Spectrum.spt_maxima(y, thr)
# #         return Spectrum.spt_fit(x, y, ispt, midx, wdw, fwhm0, gain, sat)

# #     @staticmethod
# #     def _analyze_spt2(x, y, ispt, medflt, thr, prm, dst, clst, sat, wdw, fwhm0, gain):
# #         y, _ = Spectrum.spt_filter(y, medflt)
# #         midx = Spectrum.spt_maxima2(y, thr, prm, dst, clst, sat)
# #         return Spectrum.spt_fit2(x, y, ispt, midx, wdw, fwhm0, gain, sat)

# #     def _merged_coordinates(self, ispt):
# #         with self.lfile.read() as f:
# #             crd = f['data']['coordinates'][:,:]
# #         crd = crd[ispt]
# #         gidx = np.unique(crd[:,4])
# #         if len(gidx) > 1:
# #             raise RuntimeError('Bad coordinates grouping')
# #         info = self.lfile.info[self.name]
# #         if info.areas > 1:
# #             gmap = info.gmap
# #             crd[:,0] += gmap[crd[:,3],1] * info.scan_area[0]
# #             crd[:,1] += gmap[crd[:,3],0] * info.scan_area[1]
# #         return pd.DataFrame(crd[:,:3], columns=['i','j','k'], dtype=int)

# #     def pks_hash(self):
# #         return self._pks_hash(self.pks)

# #     @staticmethod
# #     def _pks_hash(pks):
# #         return ((pks.ispt.values.astype(np.uint64) << 32) +
# #                 pks.wl.values.view(np.uint32).astype(np.uint64))

# #     # ---------------------------------------------------------------------------
# #     # New robust clustering: find_lines_and_multi
# #     # ---------------------------------------------------------------------------
# #     def find_lines_and_multi(self,
# #                              spatial_radius: float = 3.0,
# #                              merge_radius:   float = 12.0,
# #                              bic_max_comp:   int   = 8):
# #         pks = self.pks.copy()
# #         pks.index.name = None
# #         if pks is None or len(pks) == 0:
# #             self.lns = self._empty_lns()
# #             self.mlt = self._empty_mlt()
# #             return

# #         # 1) spatial particles
# #         coords = pks[['i','j','k']].to_numpy(float)
# #         Gp = radius_neighbors_graph(coords,
# #                                     radius=spatial_radius,
# #                                     include_self=True,
# #                                     n_jobs=-1)
# #         _, pid = connected_components(Gp, directed=False)
# #         pks['pid'] = pid.astype(np.uint32)

# #         # 2) energy GMM per pid -> lid
# #         next_lid = 0
# #         lids = np.empty(len(pks), dtype=int)
# #         for _, subset in pks.groupby('pid'):
# #             idx = subset.index
# #             n_peaks = len(idx)
# #             if n_peaks == 1:
# #                 lids[idx] = next_lid; next_lid += 1; continue
# #             Es = subset['E'].values.reshape(-1,1)
# #             if np.nanstd(Es) < 1e-6:
# #                 lids[idx] = next_lid; next_lid += 1; continue
# #             best_bic = np.inf
# #             best_lbl = np.zeros(n_peaks, dtype=int)
# #             max_comp = min(bic_max_comp, n_peaks)
# #             for n in range(1, max_comp+1):
# #                 if n > 1 and n_peaks < 2*n:
# #                     continue
# #                 gm = GaussianMixture(n, covariance_type='diag',
# #                                      reg_covar=1e-3,
# #                                      random_state=0)
# #                 gm.fit(Es)
# #                 bic = gm.bic(Es)
# #                 lbl = (gm.predict(Es) if n>1 else np.zeros(n_peaks,dtype=int))
# #                 if bic < best_bic:
# #                     best_bic, best_lbl = bic, lbl
# #             lids[idx] = best_lbl + next_lid
# #             next_lid += int(best_lbl.max()) + 1
# #         pks['lid'] = lids.astype(np.uint64)

# #         # 3) build lines
# #         lns = (pks.groupby('lid')
# #                .agg(i=('i','mean'), j=('j','mean'), k=('k','mean'),
# #                     a_sum=('a','sum'), E=('E','mean'), dE=('E','std'),
# #                     n=('a','count')))
# #         lns.index.name = 'lid'
# #         lns = lns.astype({'i':'float32','j':'float32','k':'float32',
# #                           'a_sum':'float32','E':'float32',
# #                           'dE':'float32','n':'int32'})

# #         # 4) merge fragments -> mid
# #         cent = lns[['i','j','k']].to_numpy()
# #         if len(cent) < 2:
# #             mids = np.zeros(len(cent), dtype=np.uint64)
# #         else:
# #             Gm = radius_neighbors_graph(cent,
# #                                         radius=merge_radius,
# #                                         include_self=False,
# #                                         n_jobs=-1)
# #             rows, cols = Gm.nonzero()
# #             for u,v in zip(rows,cols):
# #                 dxyz = np.linalg.norm(cent[u]-cent[v])
# #                 dE = abs(float(lns.E.iloc[u] - lns.E.iloc[v]))
# #                 w = np.exp(-(dxyz/merge_radius)**2) * np.exp(-(dE/5.0)**2)
# #                 Gm[u,v] = w
# #             _, mids = connected_components(Gm, directed=False)
# #             mids = mids.astype(np.uint64)
# #         lns['mid'] = mids
# #         pks['mid'] = pks['lid'].map(lns['mid']).astype(np.uint64)

# #         # 5) build multiplets
# #         if lns.empty:
# #             mlt = self._empty_mlt()
# #         else:
# #             mlt = (lns.groupby('mid')
# #                    .agg(i=('i','mean'), j=('j','mean'), k=('k','mean'),
# #                         n=('n','sum')))
# #             mlt.index.name = 'mid'
# #             mlt = mlt.astype({'i':'float32','j':'float32','k':'float32','n':'int32'})

# #         # 6) save
# #         self.pks = pks
# #         self.lns = lns
# #         self.mlt = mlt

# #     @staticmethod
# #     def _empty_lns():
# #         df = pd.DataFrame(columns=COLS.MLNS).astype(COLS.MLNS)
# #         df.index.name = 'lid'; return df

# #     @staticmethod
# #     def _empty_mlt():
# #         df = pd.DataFrame(columns=COLS.MMLT).astype(COLS.MMLT)
# #         df.index.name = 'mid'; return df


# # # ------------------------------ OPTIONS CLASSES -------------------------------

# # class PeaksOpts:
# #     OPTS = {'pool': False, 'chunk': 100000}
# #     def __init__(self, **kwds):
# #         for opt,val in PeaksOpts.OPTS.items(): setattr(self, opt, kwds.get(opt, val))
# #         for opt,val in PeakFitOpts.OPTS.items(): setattr(self, opt, kwds.get(opt, val))
# #     @property
# #     def fit_options(self):
# #         return PeakFitOpts(**{k:getattr(self,k) for k in PeakFitOpts.OPTS})

# import numpy as np
# import pandas as pd

# from sklearn.neighbors import radius_neighbors_graph
# from sklearn.mixture  import GaussianMixture
# from scipy.sparse.csgraph import connected_components

# from joblib import Parallel, delayed
# from os import cpu_count
# from time import time
# from copy import deepcopy

# from .mapline import MapLine, LineSplitter
# from ..data import LaseData_Map_Confocal
# from ..data.spectrum import Spectrum, PeakFitOpts
# from ..utils.logging import logger
# from ..utils.constants import LIDS, COLS, SDTP, K_nm2meV as KE
# from ..utils.constants import LIDS, MIDS, COLS, SDTP, K_nm2meV as KE
# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Helpers
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _hash_peaks(df: pd.DataFrame) -> np.ndarray:
# #     """32‑bit spectrum‑id in high word | 32‑bit float‑view of wavelength"""
# #     return (
# #         (df.ispt.values.astype(np.uint64) << 32)
# #         + df.wl.astype(np.float32).values.view(np.uint32).astype(np.uint64)
# #     )

# # # guarantee the analyzers always yield (n_peaks, 5) arrays --------------------

# # def _reshape_row(arr):
# #     arr = np.asarray(arr)
# #     if arr.ndim == 1:
# #         return arr.reshape(1, -1)
# #     return arr

# # # ──────────────────────────────────────────────────────────────────────────────
# # class PeaksOpts:
# #     """Minimal wrapper so PeaksOpts is always available when find_peaks is
# #     first called (the original file declared it at the bottom)."""
# #     OPTS = {"pool": False, "chunk": 100_000, "method": 2}

# #     def __init__(self, **kw):
# #         for k, v in self.OPTS.items():
# #             setattr(self, k, kw.get(k, v))
# #         # inherit spectrum‑fit options
# #         for k, v in PeakFitOpts.OPTS.items():
# #             setattr(self, k, kw.get(k, v))

# #     @property
# #     def fit_options(self):
# #         return PeakFitOpts(**{k: getattr(self, k) for k in PeakFitOpts.OPTS})

# # # ──────────────────────────────────────────────────────────────────────────────
# # class LaseAnalysis_Map:
# #     FWHME0 = 0.615  # default Lorentzian FWHM at t = 0

# #     # ---------------------------------------------------------------------
# #     def __init__(self, lfile, gname, pks=None, lns=None, mlt=None):
# #         self.lfile = lfile
# #         self.name  = gname
# #         self.pks   = pks
# #         self.lns   = lns
# #         self.mlt   = mlt
# #         self.info  = deepcopy(lfile.info[self.name])

# #     # ------------------------------------------------------------------ I/O
# #     def save_analysis(self, analysis: str = "base", overwrite: bool = False):
# #         self.lfile.save_analysis(self, analysis, overwrite)

# #     def get_data(self, peaks: bool = True):
# #         if self.lns is None or self.mlt is None:
# #             raise RuntimeError("Lines & multiplets missing — run analysis first")
# #         pks = self.pks[self.pks.lid.isin(self.lns.index)] if peaks else None
# #         return LaseData_Map_Confocal(self.mlt, self.lns, pks, name=self.name)

# #     # ------------------------------------------------------------------ PEAKS
# #     def find_peaks(self, popt=None, **kw):  # type: ignore  # remove union annotation for Python compatibility
# #         logger.info("────────── Find peaks ──────────")
# #         t0 = time()
# #         popt = popt or PeaksOpts(**kw)

# #         with self.lfile.read() as f:
# #             wl_axis = f["data"]["wl_axis"][:].astype(np.float64)
# #             coords  = f["data"]["coordinates"][:, 4]
# #             spectra = f["data"]["spectra"]

# #         g_mask  = coords == self.info.id
# #         all_idx = np.where(g_mask)[0]
# #         if popt.chunk is None:
# #             slices = [all_idx]
# #         else:
# #             slices = [all_idx[i:i+popt.chunk] for i in range(0, len(all_idx), popt.chunk)]

# #         # pick analysis function ------------------------------------------------
# #         if popt.method == 1:
# #             analyzer = self._analyze_spt
# #             ana_opts = [popt.med_filter, popt.threshold, popt.window,
# #                         self.FWHME0, popt.gain, popt.saturation]
# #         else:
# #             step    = wl_axis[1]-wl_axis[0]
# #             clst    = int(1.5*popt.window/step)
# #             analyzer = self._analyze_spt2
# #             ana_opts = [popt.med_filter, popt.threshold, popt.prominence,
# #                         popt.distance, clst, popt.saturation,
# #                         popt.window, self.FWHME0, popt.gain]

# #         pool = Parallel(cpu_count()-2) if popt.pool else None
# #         frames = []
# #         for sl in slices:
# #             with self.lfile.read() as f:
# #                 Ys = f["data"]["spectra"][sl, :].astype(np.float64)
# #             if pool:
# #                 rets = pool(delayed(analyzer)(wl_axis, y, sl[i], *ana_opts)
# #                              for i, y in enumerate(Ys))
# #             else:
# #                 rets = [analyzer(wl_axis, y, sl[i], *ana_opts)
# #                         for i, y in enumerate(Ys)]
# #             rows = [_reshape_row(r) for r in rets if r is not None and len(r)]
# #             if rows:
# #                 frames.append(pd.DataFrame(np.vstack(rows), columns=COLS.FIT))

# #         if not frames:
# #             self.pks = pd.DataFrame(columns=list(COLS.MPKS))
# #             logger.info("No peaks found  [t=%.3fs]", time()-t0)
# #             return

# #         df = pd.concat(frames, ignore_index=True).dropna()
# #         df = df.astype({"a":np.float32, "wl":np.float32, "fwhm":np.float32,
# #                         "ph":np.float32, "ispt":int})
# #         # deduplicate -----------------------------------------------------------
# #         hsh = _hash_peaks(df)
# #         dup = pd.Series(hsh).duplicated()
# #         if dup.any():
# #             df = df.loc[~dup].reset_index(drop=True)

# #         # coordinates -----------------------------------------------------------
# #         with self.lfile.read() as f:
# #             xyz_all = f["data"]["coordinates"][:, :3]
# #         xyz = xyz_all[df.ispt.values]
# #         pks = pd.DataFrame(xyz, columns=["i","j","k"], dtype=int)
# #         pks = pd.concat([pks, df], axis=1)
# #         pks["lid"]   = LIDS.NON
# #         pks["E"]     = (KE / pks.wl).astype(np.float32)
# #         pks["fwhmE"] = (pks.fwhm * pks.E / pks.wl).astype(np.float32)
# #         pks.sort_values(["ispt","E"], inplace=True)
# #         pks.index = pd.Index(range(len(pks)), name="pid", dtype=np.uint64)
# #         self.pks  = pks.astype(COLS.MPKS)
# #         logger.info("Peaks: %d  [t=%.3fs]", len(pks), time()-t0)

# #     # -------------------- single‑peak fit helpers --------------------------
# #     @staticmethod
# #     def _analyze_spt(x, y, ispt, medflt, thr, wdw, fwhm0, gain, sat):
# #         y, _ = Spectrum.spt_filter(y, medflt)
# #         idx  = Spectrum.spt_maxima(y, thr)
# #         return Spectrum.spt_fit(x, y, ispt, idx, wdw, fwhm0, gain, sat)

# #     @staticmethod
# #     def _analyze_spt2(x, y, ispt, medflt, thr, prm, dst, clst, sat, wdw, fwhm0, gain):
# #         y, _ = Spectrum.spt_filter(y, medflt)
# #         idx  = Spectrum.spt_maxima2(y, thr, prm, dst, clst, sat)
# #         return Spectrum.spt_fit2(x, y, ispt, idx, wdw, fwhm0, gain, sat)

# #     # ------------------------------------------------------------------ LINES + MULTI
# #     def find_lines_and_multi(self, spatial_radius=3.0, merge_radius=12.0, bic_max_comp=8):
# #         if self.pks is None or self.pks.empty:
# #             raise RuntimeError("run find_peaks first")
# #         logger.info("────────── Cluster peaks ──────────")
# #         t0 = time()
# #         pks = self.pks.copy()
# #         xyz = pks[["i","j","k"]].to_numpy(float)
# #         G   = radius_neighbors_graph(xyz, radius=spatial_radius, include_self=True, n_jobs=-1)
# #         _, pid = connected_components(G, directed=False)
# #         pks["pid"] = pid.astype(np.uint32)

# #         lids = np.empty(len(pks), dtype=np.uint64)
# #         next_id = 0
# #         for pid_val, grp in pks.groupby("pid"):
# #             Es   = grp.E.values.reshape(-1,1)
# #             n_pk = len(Es)
# #             if n_pk == 1 or np.nanstd(Es) < 1e-6:
# #                 lids[grp.index] = next_id
# #                 next_id += 1
# #                 continue
# #             max_comp = min(bic_max_comp, n_pk)
# #             best_lbl, best_bic = None, np.inf
# #             for k in range(1, max_comp+1):
# #                 if n_pk < 2*k:  # GM fails if too few samples
# #                     continue
# #                 gm = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=0)
# #                 gm.fit(Es)
# #                 bic = gm.bic(Es)
# #                 if bic < best_bic:
# #                     best_bic = bic
# #                     best_lbl = gm.predict(Es)
# #             lids[grp.index] = best_lbl + next_id
# #             next_id += best_lbl.max() + 1
# #         pks["lid"] = lids.astype(np.uint64)

# #         # build lines -------------------------------------------------------
# #         lns = (pks.groupby("lid")
# #                   .agg(i=("i","mean"), j=("j","mean"), k=("k","mean"),
# #                        a_sum=("a","sum"), E=("E","mean"), dE=("E","std"), n=("a","count")))
# #         lns.index.name = "lid"
# #         lns = lns.astype({"i":"float32","j":"float32","k":"float32",
# #                           "a_sum":"float32","E":"float32","dE":"float32","n":"int32"})

# #         # merge fragments into multiplets ----------------------------------
# #         centres = lns[["i","j","k"]].to_numpy()
# #         if len(centres) < 2:
# #             mids = np.zeros(len(centres), dtype=np.uint64)
# #         else:
# #             Gm = radius_neighbors_graph(centres, radius=merge_radius, include_self=False, n_jobs=-1)
# #             _, mids = connected_components(Gm, directed=False)
# #             mids = mids.astype(np.uint64)
# #         lns["mid"] = mids
# #         pks["mid"] = pks.lid.map(lns.mid).astype(np.uint64)

# #         # multiplet table ---------------------------------------------------
# #         mlt = (lns.groupby("mid")
# #                   .agg(i=("i","mean"), j=("j","mean"), k=("k","mean"), n=("n","sum")))
# #         mlt.index.name = "mid"
# #         mlt = mlt.astype({"i":"float32","j":"float32","k":"float32","n":"int32"})

# #         # save
# #         self.pks, self.lns, self.mlt = pks, lns, mlt
# #         logger.info("Lines:%d  Multiplets:%d  [t=%.3fs]", len(lns), len(mlt), time()-t0)
# # import numpy as np
# # import pandas as pd
# # from pathlib import Path
# # from sklearn.neighbors import radius_neighbors_graph
# # from sklearn.mixture import GaussianMixture
# # from scipy.sparse.csgraph import connected_components
# # from joblib import Parallel, delayed
# # from os import cpu_count
# # from time import time
# # from copy import deepcopy
# # from IPython.display import display

# # from lase_analysis.files import read_lasefile
# # from .mapline import MapLine, LineSplitter
# # from ..data import LaseData_Map_Confocal
# # from ..data.spectrum import Spectrum, PeakFitOpts
# # from ..utils.logging import logger
# # from ..utils.constants import LIDS, COLS, SDTP, K_nm2meV as KE

# # def _hash_peaks(df: pd.DataFrame) -> np.ndarray:
# #     """Unique hash: 32-bit ispt in high bits + 32-bit float bits of wl"""
# #     return (
# #         (df.ispt.values.astype(np.uint64) << 32)
# #         + df.wl.astype(np.float32).values.view(np.uint32).astype(np.uint64)
# #     )


# # def _reshape_row(arr):
# #     arr = np.asarray(arr)
# #     return arr.reshape(1, -1) if arr.ndim == 1 else arr


# # class PeaksOpts:
# #     OPTS = {"pool": False, "chunk": 100_000, "method": 2}

# #     def __init__(self, **kw):
# #         for k, v in self.OPTS.items():
# #             setattr(self, k, kw.get(k, v))
# #         # inherit spectrum‑fit options
# #         for k, v in PeakFitOpts.OPTS.items():
# #             setattr(self, k, kw.get(k, v))

# #     @property
# #     def fit_options(self):
# #         return PeakFitOpts(**{k: getattr(self, k) for k in PeakFitOpts.OPTS})


# # class LaseAnalysis_Map:
# #     FWHME0 = 0.615  # default Lorentzian FWHM

# #     def __init__(self, lfile, gname, pks=None, lns=None, mlt=None):
# #         self.lfile = lfile
# #         self.name  = gname
# #         self.pks   = pks
# #         self.lns   = lns
# #         self.mlt   = mlt
# #         self.info  = deepcopy(lfile.info[self.name])

# #     def save_analysis(self, analysis: str = "base", overwrite: bool = False):
# #         self.lfile.save_analysis(self, analysis, overwrite)

# #     def get_data(self, peaks: bool = True):
# #         if self.lns is None or self.mlt is None:
# #             raise RuntimeError("Lines & multiplets missing — run analysis first")
# #         pks = self.pks[self.pks.lid.isin(self.lns.index)] if peaks else None
# #         return LaseData_Map_Confocal(self.mlt, self.lns, pks, name=self.name)

# #     # def find_peaks(self, popt=None, **kw):
# #     #     logger.info("────────── Find peaks ──────────")
# #     #     t0 = time()
# #     #     popt = popt or PeaksOpts(**kw)

# #     #     with self.lfile.read() as f:
# #     #         wl_axis = f["data"]["wl_axis"][:].astype(np.float64)
# #     #         coords   = f["data"]["coordinates"][:, 4]
# #     #         spectra  = f["data"]["spectra"]

# #     #     mask = coords == self.info.id
# #     #     idxs = np.where(mask)[0]
# #     #     slices = [idxs] if popt.chunk is None else [idxs[i:i+popt.chunk]
# #     #                                       for i in range(0, len(idxs), popt.chunk)]

# #     #     # choose analyzer & options
# #     #     if popt.method == 1:
# #     #         analyzer = self._analyze_spt
# #     #         opts     = [popt.med_filter, popt.threshold, popt.window,
# #     #                     self.FWHME0, popt.gain, popt.saturation]
# #     #     else:
# #     #         step     = wl_axis[1] - wl_axis[0]
# #     #         clst     = int(1.5*popt.window/step)
# #     #         analyzer = self._analyze_spt2
# #     #         opts     = [popt.med_filter, popt.threshold, popt.prominence,
# #     #                     popt.distance, clst, popt.saturation,
# #     #                     popt.window, self.FWHME0, popt.gain]

# #     #     pool = Parallel(cpu_count()-2) if popt.pool else None
# #     #     frames = []
# #     #     for sl in slices:
# #     #         with self.lfile.read() as f:
# #     #             Ys = f["data"]["spectra"][sl, :].astype(np.float64)
# #     #         if pool:
# #     #             rets = pool(delayed(analyzer)(wl_axis, y, sl[i], *opts)
# #     #                          for i, y in enumerate(Ys))
# #     #         else:
# #     #             rets = [analyzer(wl_axis, y, sl[i], *opts) 
# #     #                     for i, y in enumerate(Ys)]
# #     #         rows = [_reshape_row(r) for r in rets if r is not None and len(r)]
# #     #         if rows:
# #     #             frames.append(pd.DataFrame(np.vstack(rows), columns=COLS.FIT))

# #     #     if not frames:
# #     #         self.pks = pd.DataFrame(columns=list(COLS.MPKS))
# #     #         logger.info("No peaks found  [t=%.3fs]", time()-t0)
# #     #         return

# #     #     df = pd.concat(frames, ignore_index=True).dropna()
# #     #     df = df.astype({"a":np.float32, "wl":np.float32,
# #     #                     "fwhm":np.float32, "ph":np.float32,
# #     #                     "ispt":int})

# #     #     # remove duplicates
# #     #     hsh = _hash_peaks(df)
# #     #     df = df.loc[~pd.Series(hsh).duplicated()].reset_index(drop=True)

# #     #     # map back to spatial coords
# #     #     with self.lfile.read() as f:
# #     #         xyz_all = f["data"]["coordinates"][:, :3]
# #     #     xyz = xyz_all[df.ispt.values]
# #     #     pks = pd.DataFrame(xyz, columns=["i","j","k"])
# #     #     pks = pd.concat([pks, df], axis=1)
# #     #     pks["lid"]    = LIDS.NON
# #     #     pks["E"]      = (KE / pks.wl).astype(np.float32)
# #     #     pks["fwhmE"]  = (pks.fwhm * pks.E / pks.wl).astype(np.float32)

# #     #     # drop any invalid coordinates before casting to int
# #     #     pks = pks.replace([np.inf, -np.inf], np.nan)
# #     #     pks = pks.dropna(subset=["i","j","k"]).reset_index(drop=True)

# #     #     pks = pks.sort_values(["ispt","E"]).reset_index(drop=True)
# #     #     pks.index = pd.Index(range(len(pks)), name="pid", dtype=np.uint64)
# #     #     self.pks = pks.astype(COLS.MPKS)
# #     #     logger.info("Peaks: %d  [t=%.3fs]", len(self.pks), time()-t0)
# #     def find_peaks(self, popt=None, **kw):
# #       logger.info("────────── Find peaks ──────────")
# #       t0 = time()
# #       popt = popt or PeaksOpts(**kw)

# #       with self.lfile.read() as f:
# #           wl_axis = f["data"]["wl_axis"][:].astype(np.float64)
# #           coords  = f["data"]["coordinates"][:, 4]

# #       idxs = np.where(coords == self.info.id)[0]
# #       chunks = [idxs] if popt.chunk is None else [
# #           idxs[i:i + popt.chunk] for i in range(0, len(idxs), popt.chunk)
# #       ]

# #       # choose analyser / options …
# #       # (same as before) --------------------------------------------------
# #           #     # choose analyzer & options
# #       if popt.method == 1:
# #           analyzer = self._analyze_spt
# #           opts     = [popt.med_filter, popt.threshold, popt.window,
# #                       self.FWHME0, popt.gain, popt.saturation]
# #       else:
# #           step     = wl_axis[1] - wl_axis[0]
# #           clst     = int(1.5*popt.window/step)
# #           analyzer = self._analyze_spt2
# #           opts     = [popt.med_filter, popt.threshold, popt.prominence,
# #                       popt.distance, clst, popt.saturation,
# #                       popt.window, self.FWHME0, popt.gain]

# #       pool   = Parallel(cpu_count() - 2) if popt.pool else None
# #       frames = []
# #       for sl in chunks:
# #           with self.lfile.read() as f:
# #               Ys = f["data"]["spectra"][sl].astype(np.float64)
# #           if pool:
# #               rets = pool(delayed(analyzer)(wl_axis, y, sl[i], *opts)
# #                           for i, y in enumerate(Ys))
# #           else:
# #               rets = [analyzer(wl_axis, y, sl[i], *opts)
# #                       for i, y in enumerate(Ys)]
# #           rows = [_reshape_row(r) for r in rets if r is not None and len(r)]
# #           if rows:
# #               frames.append(pd.DataFrame(np.vstack(rows), columns=COLS.FIT))

# #       if not frames:
# #           self.pks = pd.DataFrame(columns=list(COLS.MPKS))
# #           logger.info("No peaks found  [t=%.3fs]", time() - t0)
# #           return

# #       # ---------- CLEAN TABLE AND DE-DUP ---------------------------------
# #       df = (pd.concat(frames, ignore_index=True)
# #               .dropna()
# #               .reset_index(drop=True)
# #               .astype({"a": np.float32, "wl": np.float32,
# #                       "fwhm": np.float32, "ph": np.float32,
# #                       "ispt": int}))

# #       dup_mask = pd.Series(_hash_peaks(df)).duplicated()
# #       if dup_mask.any():
# #           df = df.loc[~dup_mask].reset_index(drop=True)

# #       # ---------- ADD COORDINATES & METADATA -----------------------------
# #       with self.lfile.read() as f:
# #           xyz_all = f["data"]["coordinates"][:, :3]
# #       xyz = xyz_all[df.ispt.values]

# #       pks = pd.concat([pd.DataFrame(xyz, columns=["i","j","k"]), df], axis=1)
# #       pks["lid"]   = LIDS.NON
# #       pks["E"]     = (KE / pks.wl).astype(np.float32)
# #       pks["fwhmE"] = (pks.fwhm * pks.E / pks.wl).astype(np.float32)

# #       # remove any non-finite coords **before** casting to int
# #       pks = (pks.replace([np.inf, -np.inf], np.nan)
# #                 .dropna(subset=["i", "j", "k"])
# #                 .reset_index(drop=True))

# #       pks.sort_values(["ispt", "E"], inplace=True)
# #       pks.index = pd.Index(range(len(pks)), name="pid", dtype=np.uint64)

# #       self.pks = pks.astype(COLS.MPKS)
# #       logger.info("Peaks: %d  [t=%.3fs]", len(self.pks), time() - t0)


# #     @staticmethod
# #     def _analyze_spt(x, y, ispt, medflt, thr, wdw, fwhm0, gain, sat):
# #         y, _ = Spectrum.spt_filter(y, medflt)
# #         idx  = Spectrum.spt_maxima(y, thr)
# #         return Spectrum.spt_fit(x, y, ispt, idx, wdw, fwhm0, gain, sat)

# #     @staticmethod
# #     def _analyze_spt2(x, y, ispt, medflt, thr, prm, dst, clst, sat, wdw, fwhm0, gain):
# #         y, _ = Spectrum.spt_filter(y, medflt)
# #         idx  = Spectrum.spt_maxima2(y, thr, prm, dst, clst, sat)
# #         return Spectrum.spt_fit2(x, y, ispt, idx, wdw, fwhm0, gain, sat)

# #     # def find_lines_and_multi(self, spatial_radius=3.0, merge_radius=12.0, bic_max_comp=8):
# #     #     if self.pks is None or self.pks.empty:
# #     #         raise RuntimeError("run find_peaks first")
# #     #     logger.info("────────── Cluster peaks ──────────")
# #     #     t0 = time()

# #     #     pks = self.pks.copy()
# #     #     xyz = pks[["i","j","k"]].to_numpy(float)
# #     #     G   = radius_neighbors_graph(xyz, radius=spatial_radius, include_self=True, n_jobs=-1)
# #     #     _, pid = connected_components(G, directed=False)
# #     #     pks["ccid"] = pid.astype(np.uint32)

# #     #     lids = np.empty(len(pks), dtype=np.uint64)
# #     #     next_id = 0
# #     #     for _, grp in pks.groupby("ccid"):
# #     #         Es = grp.E.values.reshape(-1,1)
# #     #         n_pk = len(Es)
# #     #         if n_pk == 1 or np.nanstd(Es) < 1e-6:
# #     #             lids[grp.index] = next_id
# #     #             next_id += 1
# #     #             continue
# #     #         best_lbl, best_bic = None, np.inf
# #     #         max_comp = min(bic_max_comp, n_pk)
# #     #         for k in range(1, max_comp+1):
# #     #             if n_pk < 2*k: continue
# #     #             gm = GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=0)
# #     #             gm.fit(Es)
# #     #             bic = gm.bic(Es)
# #     #             if bic < best_bic:
# #     #                 best_bic = bic
# #     #                 best_lbl = gm.predict(Es)
# #     #         lids[grp.index] = best_lbl + next_id
# #     #         next_id += best_lbl.max() + 1
# #     #     pks["lid"] = lids

# #     #     # build lines
# #     #     lns = (pks.groupby("lid")
# #     #            .agg(i=("i","mean"), j=("j","mean"), k=("k","mean"),
# #     #                 a=("a","sum"), E=("E","mean"),
# #     #                 dE=("E","std"), n=("a","count")))
# #     #     lns.index.name = "lid"
# #     #     lns = lns.astype(COLS.MLNS)

# #     #     # merge to multiplets
# #     #     centres = lns[["i","j","k"]].to_numpy()
# #     #     if len(centres) < 2:
# #     #         mids = np.zeros(len(centres), dtype=np.uint64)
# #     #     else:
# #     #         Gm = radius_neighbors_graph(centres, radius=merge_radius, include_self=False, n_jobs=-1)
# #     #         _, mids = connected_components(Gm, directed=False)
# #     #         mids = mids.astype(np.uint64)
# #     #     lns["mid"] = mids
# #     #     pks["mid"] = pks.lid.map(lns.mid)

# #     #     mlt = (lns.groupby("mid")
# #     #            .agg(i=("i","mean"), j=("j","mean"), k=("k","mean"),
# #     #                 n=("n","sum")))
# #     #     mlt.index.name = "mid"
# #     #     mlt = mlt.astype(COLS.MMLT)

# #     #     self.lns, self.mlt, self.pks = lns, mlt, pks
# #     #     logger.info("Lines: %d  Multiplets: %d  [t=%.3fs]", len(lns), len(mlt), time()-t0)
# #     def find_lines_and_multi(self,
# #                           spatial_radius: float = 3.0,
# #                           merge_radius:   float = 12.0,
# #                           bic_max_comp:   int   = 8):
# #       if self.pks is None or self.pks.empty:
# #           raise RuntimeError("run find_peaks first")

# #       logger.info("────────── Cluster peaks ──────────")
# #       t0 = time()

# #       # 1) spatial particles
# #       pks = self.pks.copy()
# #       coords = pks[['i','j','k']].to_numpy(float)
# #       G = radius_neighbors_graph(coords,
# #                                   radius=spatial_radius,
# #                                   include_self=True,
# #                                   n_jobs=-1)
# #       _, pid = connected_components(G, directed=False)
# #       pks['ccid'] = pid.astype(np.uint32)

# #       # 2) energy‐GMM per pid → lid
# #       lids = np.empty(len(pks), dtype=np.uint64)
# #       next_id = 0
# #       for _, grp in pks.groupby('ccid'):
# #           Es = grp.E.values.reshape(-1,1)
# #           n_pk = len(Es)
# #           if n_pk == 1 or np.nanstd(Es) < 1e-6:
# #               lids[grp.index] = next_id
# #               next_id += 1
# #               continue
# #           best_lbl, best_bic = None, np.inf
# #           max_comp = min(bic_max_comp, n_pk)
# #           for k in range(1, max_comp+1):
# #               if n_pk < 2*k: 
# #                   continue
# #               gm = GaussianMixture(k, covariance_type="diag",
# #                                     reg_covar=1e-3, random_state=0)
# #               gm.fit(Es)
# #               bic = gm.bic(Es)
# #               if bic < best_bic:
# #                   best_bic, best_lbl = bic, gm.predict(Es)
# #           lids[grp.index] = best_lbl + next_id
# #           next_id += best_lbl.max() + 1
# #       pks['lid'] = lids

# #       # ── 3) BUILD LINES using original weighted‐avg logic ────────────────
# #       # make sure _lines_info is the same as in your old class
# #       groups = pks.groupby('lid')
# #       lines_list = [self._lines_info(subdf, self.FWHME0) for _, subdf in groups]
# #       lns = pd.DataFrame(
# #           np.concatenate(lines_list),
# #           index=list(groups.groups),
# #           columns=[name for name,_ in SDTP.MLNS]
# #       )
# #       lns.index.name = 'lid'
# #       # cast to the exact COLS.MLNS dtypes:
# #       lns = lns.astype(COLS.MLNS)

# #       # ── 4) MERGE FRAGMENTS → mid (same as before) ───────────────────────
# #       cent = lns[['i','j','k']].to_numpy()
# #       if len(cent) < 2:
# #           mids = np.zeros(len(cent), dtype=np.uint64)
# #       else:
# #           Gm = radius_neighbors_graph(cent,
# #                                       radius=merge_radius,
# #                                       include_self=False,
# #                                       n_jobs=-1)
# #           _, mids = connected_components(Gm, directed=False)
# #           mids = mids.astype(np.uint64)

# #       lns['mid'] = mids
# #       pks['mid'] = pks.lid.map(lns.mid).astype(np.uint64)

# #       # ── 5) BUILD MULTIPLETS using original logic ────────────────────────
# #       m_groups = lns.groupby('mid')
# #       multi_list = [self._multi_info(subdf) for _, subdf in m_groups]
# #       mlt = pd.DataFrame(
# #           np.concatenate(multi_list),
# #           index=list(m_groups.groups),
# #           columns=[name for name,_ in SDTP.MMLT]
# #       )
# #       mlt.index.name = 'mid'
# #       mlt = mlt.astype(COLS.MMLT)

# #       # ── 6) SAVE BACK ────────────────────────────────────────────────────
# #       self.lns, self.mlt, self.pks = lns, mlt, pks
# #       logger.info("Lines: %d  Multiplets: %d  [t=%.3fs]",
# #                   len(lns), len(mlt), time()-t0)

# #     @staticmethod
# #     def _lines_info(pks, fwhmE0):
# #         E0 = np.average(pks.E,
# #                         weights=1/np.power(0.1 + np.abs(pks.fwhmE - fwhmE0), 2))
# #         return np.array([(
# #             np.average(pks.i, weights=pks.a).astype(np.float32),
# #             np.average(pks.j, weights=pks.a).astype(np.float32),
# #             np.average(pks.k, weights=pks.a).astype(np.float32),
# #             pks.a.sum().astype(np.float32),
# #             (KE/E0).astype(np.float32),
# #             np.std(pks.wl.values).astype(np.float32),
# #             E0.astype(np.float32),
# #             np.std(pks.E.values).astype(np.float32),
# #             pks.ph.sum().astype(np.float32),
# #             pks.shape[0],
# #             (pks.i.max()-pks.i.min() + pks.j.max()-pks.j.min()).astype(np.float32),
# #             MIDS.NON
# #         )], dtype=SDTP.MLNS)

# #     @staticmethod
# #     def _multi_info(lns):
# #         return np.array([(
# #             np.float32(lns.i.mean()),
# #             np.float32(lns.j.mean()),
# #             np.float32(lns.k.mean()),
# #             len(lns.index)
# #         )], dtype=SDTP.MMLT)

# # # ──────────────────────────────────────────────────────────────────────────────
# # #  The rest of the original file (options, etc.) can stay unchanged if you need
# # #  them elsewhere, but is not required by the new pipeline.


# #     @staticmethod
# #     def _empty_lns():
# #         df = pd.DataFrame(columns=COLS.MLNS).astype(COLS.MLNS)
# #         df.index.name = 'lid'; return df

# #     @staticmethod
# #     def _empty_mlt():
# #         df = pd.DataFrame(columns=COLS.MMLT).astype(COLS.MMLT)
# #         df.index.name = 'mid'; return df

















# # below is the rechceking version for new chat
# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from time import time
# from os import cpu_count
# from joblib import Parallel, delayed
# from sklearn.neighbors import radius_neighbors_graph
# from sklearn.mixture import GaussianMixture
# from scipy.sparse.csgraph import connected_components

# from .mapline import MapLine, LineSplitter
# from ..data import LaseData_Map_Confocal
# from ..data.spectrum import Spectrum, PeakFitOpts
# from ..utils.logging import logger
# from ..utils.constants import LIDS, MIDS, COLS, SDTP, K_nm2meV as KE


# def _hash_peaks(df: pd.DataFrame) -> np.ndarray:
#     """Unique hash: 32-bit ispt in high bits + 32-bit float bits of wl"""
#     return (
#         (df.ispt.values.astype(np.uint64) << 32)
#         + df.wl.astype(np.float32).values.view(np.uint32).astype(np.uint64)
#     )


# def _reshape_row(arr):
#     arr = np.asarray(arr)
#     return arr.reshape(1, -1) if arr.ndim == 1 else arr


# class PeaksOpts:
#     OPTS = {"pool": False, "chunk": 100_000, "method": 2}

#     def __init__(self, **kw):
#         for k, v in self.OPTS.items():
#             setattr(self, k, kw.get(k, v))
#         # inherit spectrum-fit options
#         for k, v in PeakFitOpts.OPTS.items():
#             setattr(self, k, kw.get(k, v))

#     @property
#     def fit_options(self):
#         return PeakFitOpts(**{k: getattr(self, k) for k in PeakFitOpts.OPTS})


# class LaseAnalysis_Map:
#     FWHME0 = 0.615  # default Lorentzian FWHM

#     def __init__(self, lfile, gname, pks=None, lns=None, mlt=None):
#         self.lfile = lfile
#         self.name  = gname
#         self.pks   = pks
#         self.lns   = lns
#         self.mlt   = mlt
#         self.info  = deepcopy(lfile.info[self.name])

#     def save_analysis(self, analysis: str = "base", overwrite: bool = False):
#         self.lfile.save_analysis(self, analysis, overwrite)

#     def get_data(self, peaks: bool = True):
#         if self.lns is None or self.mlt is None:
#             raise RuntimeError("Lines & multiplets missing — run analysis first")
#         # no boolean & here—.isin returns a boolean mask
#         pks = self.pks[self.pks.lid.isin(self.lns.index)] if peaks else None
#         return LaseData_Map_Confocal(self.mlt, self.lns, pks, name=self.name)

#     def find_peaks(self, popt=None, **kw):
#         logger.info("────────── Find peaks ──────────")
#         t0 = time()
#         popt = popt or PeaksOpts(**kw)

#         with self.lfile.read() as f:
#             wl_axis = f["data"]["wl_axis"][:].astype(np.float64)
#             coords  = f["data"]["coordinates"][:, 4]

#         # get all point IDs for this group
#         idxs = np.where(coords == self.info.id)[0]
#         if popt.chunk is None:
#             chunks = [idxs]
#         else:
#             chunks = [
#                 idxs[i : i + popt.chunk]
#                 for i in range(0, len(idxs), popt.chunk)
#             ]

#         # pick analyzer + its args
#         if popt.method == 1:
#             analyzer = self._analyze_spt
#             opts = [
#                 popt.med_filter, popt.threshold, popt.window,
#                 self.FWHME0, popt.gain, popt.saturation
#             ]
#         else:
#             step = wl_axis[1] - wl_axis[0]
#             clst = int(1.5 * popt.window / step)
#             analyzer = self._analyze_spt2
#             opts = [
#                 popt.med_filter, popt.threshold, popt.prominence,
#                 popt.distance, clst, popt.saturation,
#                 popt.window, self.FWHME0, popt.gain
#             ]

#         pool = Parallel(cpu_count() - 2) if popt.pool else None
#         frames = []

#         for sl in chunks:
#             with self.lfile.read() as f:
#                 Ys = f["data"]["spectra"][sl].astype(np.float64)

#             if pool:
#                 rets = pool(
#                     delayed(analyzer)(wl_axis, y, sl[i], *opts)
#                     for i, y in enumerate(Ys)
#                 )
#             else:
#                 rets = [
#                     analyzer(wl_axis, y, sl[i], *opts)
#                     for i, y in enumerate(Ys)
#                 ]

#             rows = [ _reshape_row(r) for r in rets if r is not None and len(r) ]
#             if rows:
#                 frames.append(pd.DataFrame(np.vstack(rows), columns=COLS.FIT))

#         if not frames:
#             self.pks = pd.DataFrame(columns=list(COLS.MPKS))
#             logger.info("No peaks found  [t=%.3fs]", time() - t0)
#             return

#         # ── concatenate, drop NaN, reset idx ─────────────────────────────
#         df = (
#             pd.concat(frames, ignore_index=True)
#               .dropna()
#               .reset_index(drop=True)
#               .astype({
#                   "a":    np.float32,
#                   "wl":   np.float32,
#                   "fwhm": np.float32,
#                   "ph":   np.float32,
#                   "ispt": int
#               })
#         )

#         # ── de-duplicate using a clean boolean mask ───────────────────────
#         dup_mask = pd.Series(_hash_peaks(df)).duplicated()
#         if dup_mask.any():
#             df = df.loc[~dup_mask].reset_index(drop=True)

#         # ── map back to spatial coords & build `pks` ────────────────────
#         with self.lfile.read() as f:
#             xyz_all = f["data"]["coordinates"][:, :3]
#         xyz = xyz_all[df.ispt.values]

#         pks = pd.concat([pd.DataFrame(xyz, columns=["i","j","k"]), df], axis=1)
#         pks["lid"]   = LIDS.NON
#         pks["E"]     = (KE / pks.wl).astype(np.float32)
#         pks["fwhmE"] = (pks.fwhm * pks.E / pks.wl).astype(np.float32)

#         # ── drop any infinities before casting int ──────────────────────
#         pks = (
#             pks.replace([np.inf, -np.inf], np.nan)
#                .dropna(subset=["i","j","k"])
#                .reset_index(drop=True)
#         )

#         pks.sort_values(["ispt", "E"], inplace=True)
#         pks.index = pd.Index(range(len(pks)), name="pid", dtype=np.uint64)
#         self.pks  = pks.astype(COLS.MPKS)

#         logger.info("Peaks: %d  [t=%.3fs]", len(self.pks), time() - t0)

#     @staticmethod
#     def _analyze_spt(x, y, ispt, medflt, thr, wdw, fwhm0, gain, sat):
#         y, _ = Spectrum.spt_filter(y, medflt)
#         idx  = Spectrum.spt_maxima(y, thr)
#         return Spectrum.spt_fit(x, y, ispt, idx, wdw, fwhm0, gain, sat)

#     @staticmethod
#     def _analyze_spt2(x, y, ispt, medflt, thr, prm, dst, clst, sat, wdw, fwhm0, gain):
#         y, _ = Spectrum.spt_filter(y, medflt)
#         idx  = Spectrum.spt_maxima2(y, thr, prm, dst, clst, sat)
#         return Spectrum.spt_fit2(x, y, ispt, idx, wdw, fwhm0, gain, sat)

#     def find_lines_and_multi(self,
#                              spatial_radius: float = 3.0,
#                              merge_radius:   float = 12.0,
#                              bic_max_comp:   int   = 8):
#         if self.pks is None or self.pks.empty:
#             raise RuntimeError("run find_peaks first")

#         logger.info("────────── Cluster peaks ──────────")
#         t0 = time()

#         # 1) initial spatial clustering ⟶ `ccid`
#         pks = self.pks.copy()
#         xyz = pks[["i","j","k"]].to_numpy(float)
#         G   = radius_neighbors_graph(xyz,
#                                      radius=spatial_radius,
#                                      include_self=True,
#                                      n_jobs=-1)
#         _, ccid = connected_components(G, directed=False)
#         pks["ccid"] = ccid.astype(np.uint32)

#         # 2) per-ccid GMM over E to assign `lid`
#         lids = np.empty(len(pks), dtype=np.uint64)
#         next_id = 0
#         for _, grp in pks.groupby("ccid"):
#             Es   = grp.E.values.reshape(-1,1)
#             n_pk = len(Es)
#             if n_pk == 1 or np.nanstd(Es) < 1e-6:
#                 lids[grp.index] = next_id
#                 next_id += 1
#                 continue
#             best_lbl, best_bic = None, np.inf
#             max_comp = min(bic_max_comp, n_pk)
#             for k in range(1, max_comp+1):
#                 if n_pk < 2*k: 
#                     continue
#                 gm = GaussianMixture(k, covariance_type="diag",
#                                      reg_covar=1e-3, random_state=0)
#                 gm.fit(Es)
#                 bic = gm.bic(Es)
#                 if bic < best_bic:
#                     best_bic, best_lbl = bic, gm.predict(Es)
#             lids[grp.index] = best_lbl + next_id
#             next_id += best_lbl.max() + 1
#         pks["lid"] = lids

#         # 3) build `lns` exactly as original weighted logic
#         groups    = pks.groupby("lid")
#         lines_arr = [ self._lines_info(subdf, self.FWHME0)
#                       for _, subdf in groups ]
#         lns = pd.DataFrame(
#             np.concatenate(lines_arr),
#             index=list(groups.groups),
#             columns=[name for name, _ in SDTP.MLNS]
#         )
#         lns.index.name = "lid"
#         lns = lns.astype(COLS.MLNS)

#         # 4) merge fragments ⟶ `mid`
#         centres = lns[["i","j","k"]].to_numpy()
#         if len(centres) < 2:
#             mids = np.zeros(len(centres), dtype=np.uint64)
#         else:
#             Gm = radius_neighbors_graph(centres,
#                                         radius=merge_radius,
#                                         include_self=False,
#                                         n_jobs=-1)
#             _, mids = connected_components(Gm, directed=False)
#             mids = mids.astype(np.uint64)
#         lns["mid"]     = mids
#         pks["mid"]     = pks.lid.map(lns.mid).astype(np.uint64)

#         # 5) build `mlt` exactly as original logic
#         m_groups    = lns.groupby("mid")
#         multi_arr   = [ self._multi_info(subdf)
#                         for _, subdf in m_groups ]
#         mlt = pd.DataFrame(
#             np.concatenate(multi_arr),
#             index=list(m_groups.groups),
#             columns=[name for name, _ in SDTP.MMLT]
#         )
#         mlt.index.name = "mid"
#         mlt = mlt.astype(COLS.MMLT)

#         # 6) save back
#         self.lns, self.mlt, self.pks = lns, mlt, pks
#         logger.info(
#             "Lines: %d  Multiplets: %d  [t=%.3fs]",
#             len(lns), len(mlt), time()-t0
#         )

#     @staticmethod
#     def _lines_info(pks, fwhmE0):
#         E0 = np.average(
#             pks.E,
#             weights=1/np.power(0.1 + np.abs(pks.fwhmE - fwhmE0), 2)
#         )
#         return np.array([(
#             np.average(pks.i, weights=pks.a).astype(np.float32),
#             np.average(pks.j, weights=pks.a).astype(np.float32),
#             np.average(pks.k, weights=pks.a).astype(np.float32),
#             pks.a.sum().astype(np.float32),
#             (KE/E0).astype(np.float32),
#             np.std(pks.wl.values).astype(np.float32),
#             E0.astype(np.float32),
#             np.std(pks.E.values).astype(np.float32),
#             pks.ph.sum().astype(np.float32),
#             pks.shape[0],
#             (pks.i.max()-pks.i.min() + pks.j.max()-pks.j.min())
#                .astype(np.float32),
#             MIDS.NON
#         )], dtype=SDTP.MLNS)

#     @staticmethod
#     def _multi_info(lns):
#         return np.array([(
#             np.float32(lns.i.mean()),
#             np.float32(lns.j.mean()),
#             np.float32(lns.k.mean()),
#             len(lns.index)
#         )], dtype=SDTP.MMLT)

#     @staticmethod
#     def _empty_lns():
#         df = pd.DataFrame(columns=COLS.MLNS).astype(COLS.MLNS)
#         df.index.name = "lid"
#         return df

#     @staticmethod
#     def _empty_mlt():
#         df = pd.DataFrame(columns=COLS.MMLT).astype(COLS.MMLT)
#         df.index.name = "mid"
#         return df

# helper options
class PeaksOpts:
    OPTS = {'pool': False, 'chunk': 100000}
    def __init__(self, **kwds):
        for o,v in PeaksOpts.OPTS.items(): setattr(self, o, kwds.get(o, v))
        for o,v in PeakFitOpts.OPTS.items(): setattr(self, o, kwds.get(o, v))

    @property
    def fit_options(self):
        return PeakFitOpts(**{k:getattr(self,k) for k in PeakFitOpts.OPTS})
