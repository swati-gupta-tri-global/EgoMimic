import os, cv2, shutil
import h5py
import json
import argparse
from robomimic.scripts.split_train_val import split_train_val_from_hdf5


parser = argparse.ArgumentParser()
parser.add_argument('hdf5_path', type=str)
args = parser.parse_args()

def prep_for_mimicplay(hdf5_path):
    target_path = hdf5_path.replace(".hdf5", "Mimicplay.hdf5")
    shutil.copy(hdf5_path, target_path)
    # os.makedirs(target_path, exist_ok=True)
    h5py_file = h5py.File(target_path, "r+")
    # import pdb; pdb.set_trace()
    # h5py_file.create_group("data/env_args")
    h5py_file["data"].attrs["env_args"] = json.dumps({})
    # h5py_file[""]

    # set num samples for each demo in data
    for k in h5py_file["data"].keys():
        # breakpoint()
        h5py_file["data"][k].attrs["num_samples"] = h5py_file["data"][k]["actions"].shape[0]

        # It seems like robomimic wants (..., C, H, W) instead of (..., H, W, C)
        # for im_number in range(1, 3):
        #     im1 = h5py_file[f"data/{k}/obs/front_image_{im_number}"][...].transpose((0, 3, 1, 2))
        #     del h5py_file[f"data/{k}/obs/front_image_{im_number}"]
        #     dset = h5py_file.create_dataset(f"data/{k}/obs/front_image_{im_number}", data=im1)


    # NOTE: REMOVE LATER, this is just so there's more than 1 demo
    if "demo_1" not in h5py_file["data"].keys():
        h5py_file["data"]["demo_1"] = h5py_file["data"]["demo_0"]
    
    # breakpoint()
    h5py_file.close()

    # Only have 1 demo so I'm going to spoof a second demo and use val ratio of 0.5
    # TODO: fix this ratio
    split_train_val_from_hdf5(hdf5_path=target_path, val_ratio=0.5, filter_key=None)



if __name__ == '__main__':
    prep_for_mimicplay(args.hdf5_path)