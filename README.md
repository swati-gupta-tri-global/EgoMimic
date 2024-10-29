# Egoplay
## Installation

```
git clone --recursive git@github.com:SimarKareer/EgoPlay.git
cd EgoPlay
conda env create -f environment.yaml
pip install -e external/robomimic
pip install -e .
python external/robomimic/robomimic/scripts/setup_macros.py
```

Set `git config --global submodule.recurse true` if you want `git pull` to automatically update the submodule as well.

Then go to  `external/robomimic/robomimic/macros_private.py` and manually add your wandb username. Make sure you have ran `wandb login` too.

-------
## Data processing
### Aloha to Robomimic Data
On robot run
- Start ros (launch)
- `record_episodes.py`
- Move the whole folder of all epsiodes onto skynet
- cd into `mimicplay/scripts/aloha_process`
- run `python aloha_to_robomimic.py`.  This will convert the aloha joint positions to 3d EE pose relative to robot base

ex) 
```bash
python aloha_to_robomimic.py --dataset /coc/flash7/datasets/egoplay/_OBOO_ROBOT/oboov2_robot_apr16/rawAloha --arm right --out /coc/flash7/datasets/egoplay/_OBOO_ROBOT/oboov2_robot_apr16/oboov2_robot_apr16_prestacked.hdf5  --extrinsics <newest extrinsics in SimarUtils.py> --data-type robot --prestack
```


### Calibration
- cd into `mimicplay/scripts/calibrate_camera`
- Run `python calibrate_egoplay.py --h5py-path <path to hdf5 from previous section.hdf5>`
- This will output the transform matrices

### Overlays
Install SAM to `emimic` via [instructions](https://github.com/facebookresearch/segment-anything-2).  It should be possible to have both in same env

Hand overlay
```
python hand_overlay.py --dataset /coc/flash7/datasets/egoplay/_OBOO_ARIA/oboo_yellow_jun12/converted/oboo_yellow_jun12_ACTGMMCompat_masked.hdf5 --sam --debug
```

Robot Overlay
```
python robot_overlay.py --dataset /coc/flash7/datasets/egoplay/_OBOO_ROBOTWA/oboo_jul29/converted/oboo_jul29_ACTGMMCompat_masked.hdf5 --arm right --extrinsics ariaJul29R
```


## Training Policies via Pytorch Lightning
EgoMimic Training (Toy in Bowl Task)
```
python scripts/pl_train.py --config configs/egomimic_oboo.json --debug
```

ACT Baseline Training
```
python scripts/pl_train.py --config configs/act.json --debug
```

For a detailed list of commands to run each experiment see [experiment_launch.md](./experiment_launch.md)

Use `--debug` to check that the pipeline works

Launching with pl on slurm cluster
`python scripts/pl_submit.py --config <config> --name <name> --description <description> --gpus-per-node <gpus-per-node>`

Offline Eval:
`python scripts/pl_train.py --dataset <dataset> --ckpt_path <ckpt> --eval`

## Rollout policies in the real world
Follow these instructions on the desktop connected to the real hardware.
1. Follow instructions in [EgoMimic hardware repo](https://github.com/SimarKareer/EgoMimic-Hardware)
2. Install the hardware package into the `emimic` conda env via
```
conda activate emimic
cd ~/interbotix_ws/src/EgoMimic-Hardware
pip install -e .
```
3. Rollout policy
```
cd EgoMimic/egomimic
python scripts/evaluation/eval_real --eval-path <path to>EgoPlay/trained_models_highlevel/<your model folder>/models/<your ckpt>.ckpt
```

### Single Policy Multi Dataset (Hand + Robot Data)
- `python scripts/pl_train.py --config configs/actSP.json --debug`


## Patch Notes
Dirty laundry
- Color jitter is manually implemented for ACT in _robomimic_to_act_data rather than using the ObsUtils color jitter
- hardcoded extrinsics in val_utils.py
- Added ac_key under base Algo in robomimic, I suppose this could just access the model.global_config
- I haven't tested whether aloha_to_robomimic_v2 works with highlevelGMMPretrain

- Remember: type == 0 is robot, type==1 is hand
- Now that our hdf5's have either hand or robot data, we can just specify this from the config.  So simply set config.train.data_type and config.train.data2_type to hand or robot.
- Make sure to set train.prestacked_actions = True when dataset actions are prestacked.  Hand data is prestacked and of shape (N, T, 3) bc coordinate frame changes each step.  Set seq_length to load should be 1 for this case


Dataloader should output batch with following format.  Not currently using dones or rewards
```
dict with keys:  dict_keys(['actions', 'rewards', 'dones', 'pad_mask', 'obs', 'type'])
actions: (1, 30)
rewards: (1, 1)
dones: (1, 1)
pad_mask: (1, 1)
obs: dict with keys:  dict_keys(['ee_pose', 'front_img_1', 'pad_mask'])
        ee_pose: (1, 3)
        front_img_1: (1, 480, 640, 3)
        pad_mask: (1, 1)
type: int
```
