from robomimic.utils.file_utils import create_hdf5_filter_key
import h5py

# Open the dataset
f = h5py.File("../datasets/bowlplace_robot.hdf5", "r")
demos = sorted(list(f["data"].keys()))
f.close()

# Create an 80-20 train-validation split
n_demos = len(demos)
n_train = int(0.8 * n_demos)
train_keys = demos[:n_train]
valid_keys = demos[n_train:]

# Create the filter keys
create_hdf5_filter_key(
    "../datasets/bowlplace_robot.hdf5",
    train_keys,
    "train"
)
create_hdf5_filter_key(
    "../datasets/bowlplace_robot.hdf5",
    valid_keys,
    "valid"
)