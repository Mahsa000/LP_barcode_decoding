
# ---------------------------------------------------------------------------
#  load_and_filter_data  –  GROUP‑AWARE, LOW‑MEMORY, BULLETPROOF
# ---------------------------------------------------------------------------
import re
import numpy as np
from pathlib import Path
from .files import read_lasefile
def load_raw_files(data_folder):
    """
    Loads the raw .map.lase files (un-analyzed or partially analyzed).
    Returns a list of (name_part, lase_data).
    """
    from .files import read_lasefile
    data_folder = Path(data_folder)
    out = []
    for lase_file_path in data_folder.glob("*.map.lase"):
        lase_data = read_lasefile(lase_file_path)
        # We do not rely on any 'analysis' existing yet
        name_part = lase_file_path.stem
        out.append((name_part, lase_data))
    return out
# ---------------------------------------------------------------------------
#  load_and_filter_data  –  GROUP‑AWARE, LOW‑MEMORY, BULLET‑PROOF
# ---------------------------------------------------------------------------
from pathlib import Path
import re
import numpy as np
from typing import Union, Optional, Iterable, Dict, Any

# -------------------------------------------------------------------------
def normalize_sample_id(fn: str) -> str:
    """Strip .map.lase / .lase / .map from a filename."""
    for ext in (".map.lase", ".lase", ".map"):
        if fn.endswith(ext):
            return fn[:-len(ext)]
    return fn


# -------------------------------------------------------------------------
def load_and_filter_data(
    data_folder : Union[str, Path],
    area_code   : Optional[int]           = None,
    digit_prefix: Optional[str]           = None,
    groups      : Optional[Iterable[str]] = None,   # e.g. ["grp_0"]
) -> Dict[str, Dict[str, Any]]:
    """
    Scan a folder for *.map.lase files and return just the spectra that
    belong to the requested group(s).

    Returns
    -------
    dict
        { "<sample>_<group>":
            { file, group, wax, crds, spts, pks, lns, mlt } }
    """
    folder = Path(data_folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    out: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------------
    for fpath in folder.glob("*.map.lase"):
        stem = fpath.stem
        if digit_prefix:
            m = re.search(re.escape(digit_prefix) + r"(\d+)", stem)
            base = m.group(1) if m else stem
        else:
            base = stem
        base = normalize_sample_id(base)

        lase   = read_lasefile(fpath)      # ← your helper
        wlaxis = lase.get_wlaxis()

        wanted = list(lase.info) if groups is None else [
            g for g in groups if g in lase.info
        ]
        if not wanted:
            continue

        # pull the whole coordinate table once (shape N×5)
        all_crds = lase.get_coordinates()          # cols: i,j,k,area,group_id

        # -----------------------------------------------------------------
        for grp in wanted:
            ginfo = lase.info[grp]                 # GroupInfo object
            # 1) indices of spectra whose 5th col == this group's id
            mask_grp = all_crds[:, 4] == ginfo.id
            if not mask_grp.any():
                continue
            gidx  = np.nonzero(mask_grp)[0]        # 1‑D row indices
            crdsG = all_crds[mask_grp]             # coordinates for this group

            # 2) optional area‑filter
            if area_code is not None:
                mask_area = crdsG[:, 3] == area_code
                if not mask_area.any():
                    continue
                gidx  = gidx[mask_area]
                crdsG = crdsG[mask_area]

            # 3) HDF5‑safe (ascending, unique)
            gidx = np.unique(gidx)

            # 4) pull those spectra
            spts = lase.get_spectra(idx=gidx)      # (n_spec, n_wl)

            # 5) matching analysis rows
            ldat = lase.get_data(gname=grp, analysis="base", peaks=True)
            pks  = ldat.pks[ldat.pks.ispt.isin(gidx)]
            lids = pks.lid.unique()
            lns  = ldat.lns.loc[lids] if ldat.lns is not None else None
            mids = lns.mid.unique() if lns is not None else []
            mlt  = ldat.mlt.loc[mids] if ldat.mlt is not None else None

            key = f"{base}_{grp}"
            out[key] = dict(
                file  = str(fpath),
                group = grp,
                group_name = grp,
                wax   = wlaxis,
                crds  = crdsG,
                spts  = spts,
                pks   = pks,
                lns   = lns,
                mlt   = mlt,
            )

    if not out:
        raise FileNotFoundError(
            f"No matching groups{'' if groups is None else ' '+str(list(groups))} "
            f"found in {data_folder}"
        )

    return out


