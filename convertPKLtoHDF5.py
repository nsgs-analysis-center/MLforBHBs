import pickle
import h5py
from pathlib import Path
import math
import gc

def convert_pkl_to_hdf5(pkl_path, hdf5_path, total_images, trainValTest=[70, 20, 10]):
    """
    Safely converts a massive .pkl file to .h5 without crashing your RAM.
    It generates the exact same filenames (e.g., 'train_img_0000') that your 
    dataset generator expects as keys for the database.
    """
    pkl_file = Path(pkl_path)
    if not pkl_file.exists():
        print(f"Error: {pkl_path} not found.")
        return

    # 1. Recreate the exact sequence of filenames your script used
    numTrain = math.floor(total_images * trainValTest[0] / 100)
    numVal = math.floor(total_images * trainValTest[1] / 100)
    numTest = total_images - numTrain - numVal
    
    splits = ['train', 'val', 'test']
    counts = [numTrain, numVal, numTest]
    
    expected_filenames = []
    for splitName, splitCount in zip(splits, counts):
        for imgIdx in range(splitCount):
            expected_filenames.append(f"{splitName}_img_{imgIdx:04d}")

    print(f"Starting conversion of {len(expected_filenames)} items...")
    print("This may take a while for 344 GB, but your RAM is safe!")

    # 2. Stream the pkl file into the hdf5 file one by one
    with open(pkl_file, 'rb') as f_in, h5py.File(hdf5_path, 'w') as f_out:
        for idx, filename in enumerate(expected_filenames):
            try:
                # Load just ONE item from the massive pickle file
                data_obj = pickle.load(f_in)
                
                waveforms = data_obj['waveforms']
                params = data_obj['params']
                
                indiv_waveforms = waveforms[:-1]
                tot_waveform = waveforms[-1]
                
                # Create the HDF5 group for this specific image
                grp = f_out.create_group(filename)
                
                # Save total waveform with compression!
                grp.create_dataset('tot_waveform', data=tot_waveform, compression='gzip', compression_opts=4)
                
                # Save individual waveforms
                indiv_grp = grp.create_group('indiv_waveforms')
                for i, wf in enumerate(indiv_waveforms):
                    indiv_grp.create_dataset(f'wf_{i}', data=wf, compression='gzip', compression_opts=4)
                
                # Save params
                params_grp = grp.create_group('params')
                for key, val in params.items():
                    params_grp.create_dataset(key, data=val)
                
                if idx % 10 == 0:
                    print(f"Converted {idx} / {total_images}...")
                
                # Force RAM cleanup
                del data_obj, waveforms, params, indiv_waveforms, tot_waveform
                gc.collect()
                
            except EOFError:
                print(f"\nFinished! Hit the end of the .pkl file at index {idx}.")
                break
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                break

    print(f"Conversion complete! Saved to {hdf5_path}")

if __name__ == '__main__':
    # UPDATE THESE VARIABLES FOR YOUR SPECIFIC RUN
    PKL_FILE = "imageData5/saved_waveforms.pkl"
    HDF5_FILE = "imageData5/saved_waveforms.h5"
    TOTAL_IMAGES = 610 # Put the exact number of images you generated here
    
    convert_pkl_to_hdf5(PKL_FILE, HDF5_FILE, TOTAL_IMAGES)