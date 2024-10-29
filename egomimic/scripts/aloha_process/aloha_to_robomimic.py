import h5py
import numpy as np
from aloha_robomimic_helper import single_episode_conversion
import argparse
import os
from tqdm import tqdm

# Example call: in aloha_process folder: python aloha_to_robomimic.py --dataset /coc/flash7/skareer6/calibrate_samples/ --arm left --out /coc/flash7/skareer6/calibrate_samples/ --task-name robomimic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to input hdf5 dataset",
    )
    parser.add_argument("--arm", type=str, help="which arm to convert data for")
    parser.add_argument("--out", type=str, help="path to output dataset")
    parser.add_argument("--task-name", type=str)

    args = parser.parse_args()

    pth = os.path.join(args.out, "converted_episodes")

    # before converting everything, check it all at least opens
    for file in tqdm(os.listdir(args.dataset)):
        # print("Trying to open " + file)
        if not os.path.isdir(os.path.join(args.dataset, file)):
            with h5py.File(os.path.join(args.dataset, file), "r") as f:
                pass

    for file in tqdm(os.listdir(args.dataset)):
        print("Trying to convert " + file)
        f = os.path.join(args.dataset, file)
        if os.path.isfile(f) and f.endswith(".hdf5"):
            if not os.path.isdir(pth):
                os.mkdir(pth)

            single_episode_conversion(f, args.arm, pth)

    if os.path.exists(os.path.join(args.out, args.task_name + ".hdf5")):
        os.remove(os.path.join(args.out, args.task_name + ".hdf5"))

    with h5py.File(os.path.join(args.out, args.task_name + ".hdf5"), "w") as dataset:
        for i, demo in enumerate(tqdm(os.listdir(pth))):
            f = os.path.join(pth, demo)
            with h5py.File(f, "r") as target:
                demo_group = dataset.create_group(f"demo{i}")
                # breakpoint()
                target.copy(target["/actions"], dataset[f"/demo{i}"])
                target.copy(target["/obs"], dataset[f"/demo{i}"])
                # print("Added " + f)

    print("Successful Conversion!")
