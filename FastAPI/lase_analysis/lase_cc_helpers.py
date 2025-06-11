# FILE: lase_cc_helpers.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage.measure import regionprops_table
from typing import Union, Optional, Dict, Any

from .data_loader import read_lasefile  # or wherever this lives

def normalize_sample_id(fn: str) -> str:
    for ext in (".map.lase", ".lase", ".map"):
        if fn.endswith(ext):
            return fn[:-len(ext)]
    return fn

def load_filtered_data(
    lase_path: Union[str, Path],
    group   : Optional[str] = None,
    area    : Optional[int] = None
) -> Dict[str, Any]:
    """
    Reads one .map.lase file, returns the single entry
    { "<sample>_<group>": {file, group, wax, crds, spts, pks, lns, mlt} }.
    """
    p = Path(lase_path)
    lase = read_lasefile(p)
    wlaxis = lase.get_wlaxis()
    wanted = [group] if group else list(lase.info)
    out = {}
    crds_all = lase.get_coordinates()  # N×5
    for grp in wanted:
        gi = lase.info.get(grp)
        if not gi:
            continue
        mask = crds_all[:,4] == gi.id
        if area is not None:
            mask &= (crds_all[:,3] == area)
        idxs = np.nonzero(mask)[0]
        if len(idxs)==0:
            continue
        crds = crds_all[mask]
        spts = lase.get_spectra(idx=idxs)
        ldat = lase.get_data(gname=grp, analysis="base", peaks=True)
        pks  = ldat.pks[ ldat.pks.ispt.isin(idxs) ]
        lids = pks.lid.unique()
        lns  = ldat.lns.loc[lids] if ldat.lns is not None else pd.DataFrame()
        mids = lns.mid.unique() if not lns.empty else []
        mlt  = ldat.mlt.loc[mids] if ldat.mlt is not None else pd.DataFrame()
        key = f"{normalize_sample_id(p.stem)}_{grp}"
        out[key] = dict(
            file=str(p),
            group=grp,
            wax=wlaxis,
            crds=crds,
            spts=spts,
            pks=pks,
            lns=lns,
            mlt=mlt
        )
    if not out:
        raise FileNotFoundError(f"No data for groups {wanted} in {p}")
    return out

def analyze_connected_components(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    From the filtered_data dict (one sample), build CCs and return
    { sample_key: { 'props': [...], 'pks':…, 'lns':…, 'mlt':… } }.
    """
    result = {}
    for k, v in data.items():
        pks_df = v['pks']
        spts   = v['spts']           # we will NOT store these
        lns_df = v.get('lns', pd.DataFrame())
        mlt_df = v.get('mlt', pd.DataFrame())
        i = pks_df['i'].astype(int).to_numpy()
        j = pks_df['j'].astype(int).to_numpy()
        ph= pks_df['ph'].to_numpy()
        # build image
        img = np.zeros((i.max()+1, j.max()+1), float)
        np.add.at(img, (i, j), ph)
        lab_im, n = label(img>0, structure=np.ones((3,3),int))
        props = regionprops_table(
            lab_im, intensity_image=img,
            properties=['label','area','centroid','bbox','perimeter',
                        'major_axis_length','minor_axis_length']
        )
        props_df = pd.DataFrame(props)
        pks_df['cc_label'] = lab_im[i, j]
        comps = []
        grp = k.rsplit('_',1)[-1]
        base= k[: - (len(grp)+1)]
        for _, row in props_df.iterrows():
            lbl = int(row['label'])
            if lbl==0: continue
            mask = pks_df['cc_label']==lbl
            sub = pks_df[mask]
            lids = sub.lid.unique()
            lns_sub = lns_df.loc[lids] if not lns_df.empty else pd.DataFrame()
            mids    = lns_sub.mid.unique() if not lns_sub.empty else []
            mlt_sub = mlt_df.loc[mids]    if not mlt_df.empty else pd.DataFrame()
            comps.append({
                'label':lbl,
                'area': row['area'],
                'centroid':(row['centroid-0'], row['centroid-1']),
                'bbox':tuple(row[['bbox-0','bbox-1','bbox-2','bbox-3']]),
                'perimeter':row['perimeter'],
                'major_axis_length':row['major_axis_length'],
                'minor_axis_length':row['minor_axis_length'],
                'ispt_values': sub['ispt'].to_numpy(),
                'sample_id': base,
                'group_name': grp,
                'lns':lns_sub,
                'mlt':mlt_sub
            })
        result[k] = {
            'props': comps,
            'pks': pks_df,
            'lns':lns_df,
            'mlt':mlt_df
        }
    return result

def save_cc_data(cc_data: Dict[str, Any], out_path: Union[str, Path]):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(cc_data, f)
