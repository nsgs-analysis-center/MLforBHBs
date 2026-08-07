import h5py
import numpy as np
import math
from pathlib import Path
from scipy.signal import decimate

# ---- constants matching your generator ----
DT = 2.5                                            # original sample spacing (s)
ASTRONOMICAL_YEAR = 31557600.0
T_OBS_TOT = 2 * ASTRONOMICAL_YEAR
VISIBILITY_LOG_THRESHOLD = -22.12 + math.log10(2)   # your bbox-code rule

# ---- parameter normalization (your imageData5 priors) ----
# tC is NOT here -- handled at train time as signed-log tau.
PARAM_SPECS = [
    ("logmT",    "linear", 5.5,             7.0),
    ("distance", "log10",  math.log10(5e2), math.log10(1e3)),
]
N_PARAMS = len(PARAM_SPECS)


def _normalize_params(params, i):
    out = np.empty(N_PARAMS, dtype=np.float32)
    for p, (key, tform, lo, hi) in enumerate(PARAM_SPECS):
        v = float(np.asarray(params[key])[i])
        if tform == "log10":
            v = math.log10(v)
        out[p] = (v - lo) / (hi - lo)
    return out


def _staged_decimate(x, factor):
    """Anti-aliased decimation, split into gentle stages for a cleaner filter."""
    if factor == 16:
        stages = [4, 2, 2]          # 4*2*2 = 16
    elif factor == 8:
        stages = [2, 2, 2]
    elif factor == 4:
        stages = [2, 2]
    elif factor == 1:
        return x.astype(np.float64)
    else:
        stages = [factor]           # fallback: single stage
    y = x.astype(np.float64)
    for s in stages:
        y = decimate(y, s, ftype="fir", zero_phase=True)
    return y


def _read_image(fin):
    """Return (tot_waveform, indiv_dict, params_dict) from a source file whose
    datasets live at the ROOT (one image per file)."""
    tot = np.asarray(fin["tot_waveform"][:], dtype=np.float64)
    indiv = {k: np.asarray(fin["indiv_waveforms"][k][:], dtype=np.float64)
             for k in fin["indiv_waveforms"].keys()}
    params = {k: np.asarray(fin["params"][k][:]) for k in fin["params"].keys()}
    return tot, indiv, params


def _process_image(tot, indiv, params, key, fout, decimate_factor, dt_dec):
    """Decimate, compute visible_start, normalize params, write one group to fout."""
    tC = np.asarray(params["tC"], dtype=np.float64)
    n_bhb = len(tC)

    # --- decimate the clean total stream, store as float32 (2x space win) ---
    tot_dec = _staged_decimate(tot, decimate_factor).astype(np.float32)
    L_dec = tot_dec.shape[0]

    # --- per-BHB visible_start on the PADDED individual signal ---
    # Your TDIPlacement puts the wf peak (argmax|.|) at tC.
    visible_start = np.full(n_bhb, -1, dtype=np.int64)
    for i in range(n_bhb):
        wf = indiv.get(f"wf_{i}")
        if wf is None or wf.size == 0:
            continue
        peak_idx = int(np.argmax(np.abs(wf)))          # coalescence within wf
        tC_idx = int(round(tC[i] / DT))                # placement in stream
        start_in_stream = tC_idx - peak_idx            # where wf[0] lands
        with np.errstate(divide="ignore"):
            logabs = np.log10(np.abs(wf) + 1e-300)
        above = np.where(logabs >= VISIBILITY_LOG_THRESHOLD)[0]
        if above.size == 0:
            continue
        abs_idx = max(0, start_in_stream + int(above.min()))   # original-sample coord
        visible_start[i] = abs_idx // decimate_factor          # -> decimated coord

    # --- normalized [logmT, distance] targets ---
    if n_bhb:
        norm = np.stack([_normalize_params(params, i) for i in range(n_bhb)],
                        axis=0).astype(np.float32)
    else:
        norm = np.zeros((0, N_PARAMS), np.float32)

    g = fout.create_group(key)
    g.create_dataset("waveform", data=tot_dec,
                     compression="gzip", compression_opts=4, shuffle=True)
    g.create_dataset("visible_start", data=visible_start)
    g.create_dataset("tC", data=tC.astype(np.float64))
    g.create_dataset("params", data=norm)
    g.attrs["L_dec"] = L_dec
    g.attrs["n_bhb"] = n_bhb


def build_training_h5(src_dir="imageData5/h5_out",
                      dst_path="imageData5/train_ready.h5",
                      decimate_factor=16,
                      src_glob="*.h5"):
    """Build the consolidated, decimated training cache from many per-image .h5 files.

    Source layout (one image per file, datasets at ROOT):
        <name>.h5
        ├── tot_waveform
        ├── indiv_waveforms/wf_i
        └── params/<key>
    Output: a single train_ready.h5 with one group per image (key = filename stem),
    each holding decimated 'waveform' (float32), 'visible_start', 'tC', 'params'.
    """
    dt_dec = DT * decimate_factor
    src_files = sorted(Path(src_dir).glob(src_glob))
    if not src_files:
        raise FileNotFoundError(f"No {src_glob} files in {src_dir}")

    with h5py.File(dst_path, "w") as fout:
        fout.attrs["decimate_factor"] = decimate_factor
        fout.attrs["dt"] = DT
        fout.attrs["dt_dec"] = dt_dec
        fout.attrs["T_obs_tot"] = T_OBS_TOT
        fout.attrs["vis_threshold"] = VISIBILITY_LOG_THRESHOLD
        fout.attrs["param_keys"] = [k for k, _, _, _ in PARAM_SPECS]

        n = 0
        for src in src_files:
            key = src.stem                              # e.g. 'train_img_0042'
            if key in fout:                             # skip dupes (resume-ish)
                continue
            try:
                with h5py.File(src, "r") as fin:
                    # robust: handle root-level datasets OR a single nested group
                    if "tot_waveform" in fin:
                        tot, indiv, params = _read_image(fin)
                    else:                               # fallback: one group inside
                        gkey = sorted(fin.keys())[0]
                        tot, indiv, params = _read_image(fin[gkey])
                    _process_image(tot, indiv, params, key, fout,
                                   decimate_factor, dt_dec)
            except Exception as e:
                print(f"  WARNING: failed on {src.name}: {e} -- skipping.")
                continue

            if n % 10 == 0:
                print(f"built {n}/{len(src_files)}  ({key})")
            n += 1

    print(f"done -> {dst_path}  ({n} images)")


if __name__ == "__main__":
    build_training_h5(
        src_dir="imageData5/h5_out",       # matches your OUT_DIR
        dst_path="imageData5/train_ready.h5",
        decimate_factor=16,
    )
