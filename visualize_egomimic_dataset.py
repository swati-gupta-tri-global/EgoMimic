import h5py
import numpy as np
def main(dataset_path):
    f = h5py.File(dataset_path, "r")
    print("Keys in the dataset:") # data, mask
    for key in f.keys():
        print(key)
    
    demos = list(f["data"].keys())
    num_demos = len(demos)
    print(f"Number of demonstrations: {num_demos}") # 50 demos

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    for ep in demos:
        num_actions = f["data/{}/actions".format(ep)].shape[0] # error: no actions!
        print("{} has {} samples".format(ep, num_actions))

        # print (f["data"][demos[0]])

if __name__ == "__main__":
    main(dataset_path="datasets/groceries_human.hdf5")