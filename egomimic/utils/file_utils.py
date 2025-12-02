"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.
"""

import os
import h5py
import json
import time
import urllib.request
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
from egomimic.configs import config_factory
from egomimic.algo import algo_factory
from egomimic.algo import RolloutPolicy


def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """
    Creates a new hdf5 filter key in hdf5 file @hdf5_path with
    name @key_name that corresponds to the demonstrations
    @demo_keys. Filter keys are generally useful to create
    named subsets of the demonstrations in an hdf5, making it
    easy to train, test, or report statistics on a subset of
    the trajectories in a file.

    Returns the list of episode lengths that correspond to the filtering.

    Args:
        hdf5_path (str): path to hdf5 file
        demo_keys ([str]): list of demonstration keys which should
            correspond to this filter key. For example, ["demo_0",
            "demo_1"].
        key_name (str): name of filter key to create

    Returns:
        ep_lengths ([int]): list of episode lengths that corresponds to
            each demonstration in the new filter key
    """
    f = h5py.File(hdf5_path, "a")
    demos = sorted(list(f["data"].keys()))

    # collect episode lengths for the keys of interest
    ep_lengths = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if ep in demo_keys:
            ep_lengths.append(ep_data_grp.attrs["num_samples"])

    # store list of filtered keys under mask group
    k = "mask/{}".format(key_name)
    if k in f:
        del f[k]
    f[k] = np.array(demo_keys, dtype="S")

    f.close()
    return ep_lengths


def get_env_metadata_from_dataset(dataset_path):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    return env_meta


def get_shape_metadata_from_dataset(dataset_path, all_obs_keys=None, verbose=False, ac_key="actions"):
    """
    Retrieves shape metadata from dataset(s).

    Args:
        dataset_path (str or list): path to dataset(s). Can be a single path or list of paths.
        all_obs_keys (list): list of all modalities used by the model. If not provided, all modalities
            present in the file are used.
        verbose (bool): if True, include print statements
        ac_key (str): action key to use for extracting action dimension

    Returns:
        shape_meta (dict): shape metadata. Contains the following keys:

            :`'ac_dim'`: action space dimension
            :`'all_shapes'`: dictionary that maps observation key string to shape
            :`'all_obs_keys'`: list of all observation modalities used
            :`'use_images'`: bool, whether or not image modalities are present
    """

    shape_meta = {}

    # Handle both single path and multiple paths
    if isinstance(dataset_path, str):
        dataset_paths = [dataset_path]
    else:
        dataset_paths = dataset_path

    # read demo file for some metadata - use first file as reference
    first_dataset_path = os.path.expanduser(dataset_paths[0])
    print(f'Getting shape metadata from first dataset: {first_dataset_path}')
    
    try:
        f = h5py.File(first_dataset_path, "r")
    except Exception as e:
        print(f"Error opening HDF5 file {first_dataset_path}: {e}")
        raise
        
    try:
        if "data" not in f:
            print(f"Error: No 'data' group found in {first_dataset_path}")
            print(f"Available top-level groups: {list(f.keys())}")
            raise KeyError("No 'data' group found")
            
        demo_keys = list(f["data"].keys())
        if not demo_keys:
            print(f"Error: No demos found in data group")
            raise ValueError("No demos found in data group")
            
        demo_id = demo_keys[0]
        print(f"Using demo '{demo_id}' from {len(demo_keys)} available demos")
        
        if demo_id not in f["data"]:
            print(f"Error: Demo '{demo_id}' not accessible")
            raise KeyError(f"Demo '{demo_id}' not accessible")
            
        demo = f["data/{}".format(demo_id)]
        print(f"Demo structure: {list(demo.keys())}")
        
    except Exception as e:
        print(f"Error accessing demo structure: {e}")
        f.close()
        raise

    # action dimension
    try:
        action_data = f["data/{}/{}".format(demo_id, ac_key)]
        print(f"Action data shape: {action_data.shape}")
        # Action data is shaped as [batch, interp_dim, ac_dim], so we want shape[2]
        shape_meta["ac_dim"] = action_data.shape[2] 
        print(f"Action key '{ac_key}' found with shape: {action_data.shape}")
        print(f"Action dimension (ac_dim) set to: {shape_meta['ac_dim']}")
    except KeyError:
        # Fallback to default actions key if ac_key doesn't exist
        try:
            action_data = f["data/{}/actions".format(demo_id)]
            shape_meta["ac_dim"] = action_data.shape[2]  # Use shape[2] for [batch, interp_dim, ac_dim]
            print(f"Fallback: using 'actions' key with shape: {action_data.shape}")
            print(f"Action dimension (ac_dim) set to: {shape_meta['ac_dim']}")
        except KeyError:
            print(f"Error: Neither '{ac_key}' nor 'actions' found in demo {demo_id}")
            # List available action keys for debugging
            available_keys = list(demo.keys())
            action_keys = [k for k in available_keys if 'action' in k.lower()]
            print(f"Available keys in demo: {available_keys}")
            print(f"Keys containing 'action': {action_keys}")
            raise

    # Validate action dimensions across all files if multiple files are provided
    if len(dataset_paths) > 1:
        print(f"Validating action dimensions across {len(dataset_paths)} files...")
        reference_ac_dim = shape_meta["ac_dim"]
        
        for i, dataset_path in enumerate(dataset_paths[1:], 1):
            try:
                dataset_path = os.path.expanduser(dataset_path)
                print(f"Checking file {i+1}/{len(dataset_paths)}: {dataset_path}")
                
                with h5py.File(dataset_path, "r") as f_check:
                    demo_id_check = list(f_check["data"].keys())[0]
                    
                    try:
                        action_data_check = f_check["data/{}/{}".format(demo_id_check, ac_key)]
                        file_ac_dim = action_data_check.shape[2]  # Use shape[2] for [batch, interp_dim, ac_dim]
                    except KeyError:
                        try:
                            action_data_check = f_check["data/{}/actions".format(demo_id_check)]
                            file_ac_dim = action_data_check.shape[2]  # Use shape[2] for [batch, interp_dim, ac_dim]
                        except KeyError:
                            print(f"Warning: No action data found in file {i+1}, skipping validation")
                            continue
                    
                    print(f"  File {i+1} action shape: {action_data_check.shape}, ac_dim: {file_ac_dim}")
                    
                    if file_ac_dim != reference_ac_dim:
                        print(f"ERROR: Action dimension mismatch!")
                        print(f"  Reference (file 1): {reference_ac_dim}")
                        print(f"  File {i+1}: {file_ac_dim}")
                        print(f"  This will cause issues during training!")
                        # You might want to raise an exception here or handle this case
                        
            except Exception as e:
                print(f"Error validating file {i+1}: {e}")
                continue
                
        print(f"Action dimension validation complete. Using ac_dim={reference_ac_dim}")

    # observation dimensions - process while file is still open
    all_shapes = OrderedDict()

    # Always check what obs keys are actually available in the file
    try:
        if "obs" not in demo:
            print(f"Error: No 'obs' group found in demo {demo_id}")
            print(f"Available groups in demo: {list(demo.keys())}")
            raise KeyError("No 'obs' group found")
        
        available_obs_keys = list(demo["obs"].keys())
        print(f"Available obs keys in file: {available_obs_keys}")
    except Exception as e:
        print(f"Error checking available obs keys: {e}")
        f.close()
        raise

    if all_obs_keys is None:
        # use all modalities present in the file
        all_obs_keys = available_obs_keys
        print(f"Using all available obs keys: {all_obs_keys}")
    else:
        print(f"Requested obs keys from config: {all_obs_keys}")
        # Check if requested keys exist
        missing_keys = set(all_obs_keys) - set(available_obs_keys)
        if missing_keys:
            print(f"WARNING: Requested obs keys not found in file: {missing_keys}")
            print(f"Available obs keys: {available_obs_keys}")
            # Filter to only use available keys
            all_obs_keys = [k for k in all_obs_keys if k in available_obs_keys]
            print(f"Using filtered obs keys: {all_obs_keys}")

    print(f"Processing {len(all_obs_keys)} observation keys: {all_obs_keys}")
    
    for k in sorted(all_obs_keys):
        # print(k)
        try:
            obs_path = "obs/{}".format(k)
            if obs_path not in demo:
                print(f"Warning: Observation key '{k}' not found at path '{obs_path}'")
                print(f"Available obs keys: {list(demo['obs'].keys()) if 'obs' in demo else 'No obs group'}")
                print(f"Trying alternative access methods...")
                
                # Try direct access to obs group
                if 'obs' in demo and k in demo['obs']:
                    print(f"Found '{k}' in obs group via direct access")
                    obs_data = demo['obs'][k]
                    initial_shape = obs_data.shape[1:]
                else:
                    print(f"Skipping observation key '{k}' - not accessible")
                    continue
            else:
                obs_data = demo[obs_path]
                initial_shape = obs_data.shape[1:]
                
            if verbose:
                print("obs key {} with shape {}".format(k, initial_shape))
            # Store processed shape for each obs key
            all_shapes[k] = ObsUtils.get_processed_shape(
                obs_modality=ObsUtils.OBS_KEYS_TO_MODALITIES[k],
                input_shape=initial_shape,
            )
        except Exception as e:
            print(f"Error processing observation key '{k}': {e}")
            print(f"Demo structure around obs: {list(demo.keys())}")
            if 'obs' in demo:
                print(f"Available obs keys: {list(demo['obs'].keys())}")
            f.close()
            raise

    # Close file after processing all data
    f.close()

    shape_meta["all_shapes"] = all_shapes
    shape_meta["all_obs_keys"] = all_obs_keys
    shape_meta["use_images"] = ObsUtils.has_modality("rgb", all_obs_keys)

    return shape_meta


def load_dict_from_checkpoint(ckpt_path):
    """
    Load checkpoint dictionary from a checkpoint file.

    Args:
        ckpt_path (str): Path to checkpoint file.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    if not torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    else:
        ckpt_dict = torch.load(ckpt_path)
    return ckpt_dict


def maybe_dict_from_checkpoint(ckpt_path=None, ckpt_dict=None):
    """
    Utility function for the common use case where either an ckpt path
    or a ckpt_dict is provided. This is a no-op if ckpt_dict is not
    None, otherwise it loads the model dict from the ckpt path.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

    Returns:
        ckpt_dict (dict): Loaded checkpoint dictionary.
    """
    assert (ckpt_path is not None) or (ckpt_dict is not None)
    if ckpt_dict is None:
        ckpt_dict = load_dict_from_checkpoint(ckpt_path)
    return ckpt_dict


def algo_name_from_checkpoint(ckpt_path=None, ckpt_dict=None):
    """
    Return algorithm name that was used to train a checkpoint or
    loaded model dictionary.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

    Returns:
        algo_name (str): algorithm name

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)
    algo_name = ckpt_dict["algo_name"]
    return algo_name, ckpt_dict


def config_from_checkpoint(
    algo_name=None, ckpt_path=None, ckpt_dict=None, verbose=False
):
    """
    Helper function to restore config from a checkpoint file or loaded model dictionary.

    Args:
        algo_name (str): Algorithm name.

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        config (dict): Raw loaded configuration, without properties replaced.

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)
    if algo_name is None:
        algo_name, _ = algo_name_from_checkpoint(ckpt_dict=ckpt_dict)

    if verbose:
        print("============= Loaded Config =============")
        print(ckpt_dict["config"])

    # restore config from loaded model dictionary
    config_json = ckpt_dict["config"]
    config = config_factory(algo_name, dic=json.loads(config_json))

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    return config, ckpt_dict


def policy_from_checkpoint(
    device=None, ckpt_path=None, ckpt_dict=None, verbose=False, update_obs_dict=True
):
    """
    This function restores a trained policy from a checkpoint file or
    loaded model dictionary.

    Args:
        device (torch.device): if provided, put model on this device

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        model (RolloutPolicy): instance of Algo that has the saved weights from
            the checkpoint file, and also acts as a policy that can easily
            interact with an environment in a training loop

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # algo name and config from model dict
    algo_name, _ = algo_name_from_checkpoint(ckpt_dict=ckpt_dict)

    config, _ = config_from_checkpoint(
        algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose
    )

    if update_obs_dict:
        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        ObsUtils.initialize_obs_utils_with_config(config)

    # env meta from model dict to get info needed to create model
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    if verbose:
        print("============= Loaded Policy =============")
        print(model)
    return model, ckpt_dict


def env_from_checkpoint(
    ckpt_path=None,
    ckpt_dict=None,
    env_name=None,
    render=False,
    render_offscreen=False,
    verbose=False,
    bddl_file_name=None,
):
    """
    Creates an environment using the metadata saved in a checkpoint.

    Args:
        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        env_name (str): if provided, override environment name saved in checkpoint

        render (bool): if True, environment supports on-screen rendering

        render_offscreen (bool): if True, environment supports off-screen rendering. This
            is forced to be True if saved model uses image observations.

    Returns:
        env (EnvBase instance): environment created using checkpoint

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # metadata from model dict to get info needed to create environment
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]

    if bddl_file_name is not None:
        env_meta["env_kwargs"]["bddl_file_name"] = bddl_file_name

    # create env from saved metadata
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=shape_meta["use_images"],
    )
    if verbose:
        print("============= Loaded Environment =============")
        print(env)
    return env, ckpt_dict


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.

    Args:
        url (str): url string

    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, check_overwrite=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.

    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n\n"
        )
        assert user_response.lower() in {
            "yes",
            "y",
        }, f"Did not receive confirmation. Aborting download."

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)
