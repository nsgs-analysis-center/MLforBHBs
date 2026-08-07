import pickle, h5py, os

PKL_PATH = "imageData5/saved_waveforms.pkl"
TEST_H5  = "test.h5"
N_TEST   = 5   # how many items to sample

# measure how many pkl bytes the first N items occupy
with open(PKL_PATH, "rb") as f:
    start = f.tell()
    objs = []
    for i in range(N_TEST):
        objs.append(pickle.load(f))
    pkl_bytes = f.tell() - start

# write those same items to a temp HDF5 exactly like your real loop
with h5py.File(TEST_H5, "w") as f_out:
    for idx, data_obj in enumerate(objs):
        waveforms = data_obj['waveforms']
        params    = data_obj['params']

        indiv_waveforms = waveforms[:-1]
        tot_waveform    = waveforms[-1]

        grp = f_out.create_group(f"item_{idx:04d}")
        grp.create_dataset('tot_waveform', data=tot_waveform,
                           compression='gzip', compression_opts=4)

        indiv_grp = grp.create_group('indiv_waveforms')
        for i, wf in enumerate(indiv_waveforms):
            indiv_grp.create_dataset(f'wf_{i}', data=wf,
                                     compression='gzip', compression_opts=4)

        params_grp = grp.create_group('params')
        for key, val in params.items():
            params_grp.create_dataset(key, data=val)

h5_bytes = os.path.getsize(TEST_H5)

print(f"pkl: {pkl_bytes/1e6:.1f} MB  ->  h5: {h5_bytes/1e6:.1f} MB")
print(f"ratio (h5/pkl): {h5_bytes/pkl_bytes:.3f}")
print(f"projected full h5 size: ~{(h5_bytes/pkl_bytes)*344:.0f} GB")

# clean up the test file so it doesn't eat your 10 GB
os.remove(TEST_H5)
