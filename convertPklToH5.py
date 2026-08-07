import pickle, h5py, os, gc, math
from pathlib import Path

PKL_PATH   = "imageData5/saved_waveforms.pkl"
OUT_DIR    = "imageData5/h5_out"
TOTAL      = 610
SPLIT      = [70, 20, 10]

def build_filenames(total, split):
    nTrain = math.floor(total * split[0] / 100)
    nVal   = math.floor(total * split[1] / 100)
    nTest  = total - nTrain - nVal
    names = []
    for name, cnt in zip(['train', 'val', 'test'], [nTrain, nVal, nTest]):
        names += [f"{name}_img_{i:04d}" for i in range(cnt)]
    return names

def scout(pkl_path):
    """Non-destructive: record the start byte offset of every VALID pickled object.
    Stops cleanly at EOF or at a truncated/corrupt trailing record."""
    offsets = [0]
    n = 0
    with open(pkl_path, "rb") as f:
        while True:
            try:
                pickle.load(f)
                offsets.append(f.tell())
                n += 1
                if n % 50 == 0:
                    print(f"  scouted {n} items...")
            except EOFError:
                break
            except (pickle.UnpicklingError, ValueError, OSError) as e:
                # Partial/corrupt trailing record (e.g. generation died mid-write).
                print(f"  reached a truncated/corrupt record after {n} valid items: {e}")
                print("  treating this as end-of-data; the incomplete trailing item is dropped.")
                break
    return offsets[:-1]   # drop the start-byte of the (now-skipped) trailing item

def write_item(data_obj, out_path):
    """Write one item to its own .h5 via a temp file, then fsync + atomic rename."""
    waveforms = data_obj['waveforms']
    params    = data_obj['params']
    indiv     = waveforms[:-1]
    tot       = waveforms[-1]

    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as g:
        g.create_dataset('tot_waveform', data=tot,
                         compression='gzip', compression_opts=4)
        ig = g.create_group('indiv_waveforms')
        for i, wf in enumerate(indiv):
            ig.create_dataset(f'wf_{i}', data=wf,
                              compression='gzip', compression_opts=4)
        pg = g.create_group('params')
        for k, v in params.items():
            pg.create_dataset(k, data=v)
        g.flush()

    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)  # force to disk
    os.replace(tmp, out_path)                                   # atomic

def main():
    pkl = Path(PKL_PATH)
    if not pkl.exists():
        print(f"Error: {PKL_PATH} not found."); return
    os.makedirs(OUT_DIR, exist_ok=True)

    all_names = build_filenames(TOTAL, SPLIT)

    print("Phase 1: scouting (non-destructive)...")
    offsets = scout(pkl)
    print(f"  found {len(offsets)} items in the pkl.")

    if len(offsets) > TOTAL:
        print(f"CRITICAL: pkl has {len(offsets)} items but TOTAL={TOTAL}. Aborting."); return
    # remaining items are the FIRST len(offsets) names (tail was already converted+truncated)
    names = all_names[:len(offsets)]

    print(f"\nPhase 2: converting {len(offsets)} items in reverse (pkl will shrink)...")
    with open(pkl, "r+b") as f_in:
        for idx, (offset, name) in enumerate(zip(reversed(offsets), reversed(names))):
            out_path = os.path.join(OUT_DIR, f"{name}.h5")
            try:
                if not os.path.exists(out_path):        # skip already-done (resume)
                    f_in.seek(offset)
                    data_obj = pickle.load(f_in)
                    write_item(data_obj, out_path)
                    del data_obj

                f_in.flush(); os.fsync(f_in.fileno())    # durability before chop
                f_in.truncate(offset)                    # reclaim disk space

                if idx % 10 == 0:
                    print(f"  {idx}/{len(offsets)} done ({name}), pkl shrinking...")
                gc.collect()
            except Exception as e:
                print(f"STOP at idx {idx} ({name}): {e}")
                print("Nothing past this point was truncated. Converted .h5 files are safe.")
                print("You can re-run this script to resume.")
                return

    print(f"\nDone. {len(all_names)} .h5 files in {OUT_DIR}. pkl should be ~0 bytes.")

if __name__ == '__main__':
    main()
