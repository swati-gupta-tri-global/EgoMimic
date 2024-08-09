# Egoplay
## Installation

```
git clone --recursive git@github.com:SimarKareer/EgoPlay.git
conda env create -f environment.yaml
pip install -e external/robomimic
pip install -e external/act
pip install -e external/act/detr
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
- run `python aloha_to_robomimicv2.py`.  This will convert the aloha joint positions to 3d EE pose relative to robot base

ex) 
```bash
python aloha_to_robomimicv2.py --dataset /coc/flash7/datasets/egoplay/_OBOO_ROBOT/oboov2_robot_apr16/rawAloha --arm right --out /coc/flash7/datasets/egoplay/_OBOO_ROBOT/oboov2_robot_apr16/oboov2_robot_apr16_prestacked.hdf5  --extrinsics humanoidApr16 --data-type robot --prestack
```


### Calibration
- cd into `mimicplay/scripts/calibrate_camera`
- Run `python calibrate_egoplay.py --h5py-path <path to hdf5 from previous section.hdf5>`
- This will output the transform matrices

### Overlays
Create SAM environment
```
conda env create -f sam_env.yaml
```
Then follow SAM2 install (instructions)[https://github.com/facebookresearch/segment-anything-2]

Hand overlay
```
python hand_overlay.py --hdf5_file /coc/flash7/datasets/egoplay/_OBOO_ARIA/oboo_yellow_jun12/converted/oboo_yellow_jun12_ACTGMMCompat_masked.hdf5 --debug
```

Robot Overlay
```
python robot_overlay.py --dataset /coc/flash7/datasets/egoplay/_OBOO_ROBOTWA/oboo_jul29/converted/oboo_jul29_ACTGMMCompat_masked.hdf5 --arm right --extrinsics ariaJul29R
```


## Training Policies via Pytorch Lightning
ACT Style
`python scripts/pl_train.py --config configs/act.json --dataset /coc/flash7/datasets/egoplay/_OBOO_ROBOT/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --debug`

GMM Style
`python scripts/pl_train.py --config configs/highlevel_dino_lora.json --dataset /coc/flash7/datasets/egoplay/_OBOO_ARIA/oboo_aria_apr11/rawAria/oboo_aria_apr11/converted/oboo_aria_apr11_Mimicplay_LH3.hdf5 --debug`

Launching with pl
`python scripts/pl_submit.py --config configs/act.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --name vanillaACTPL --description 8GBS32LR5e5 --num-nodes 1 --gpus-per-node 8 --batch-size 32 --lr 1e-4`

Use `--debug` to check that the pipeline works


Offline Eval:
`python scripts/pl_train.py --dataset /coc/flash7/datasets/egoplay/oboo_black/oboo_black.hdf5 --ckpt_path /coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/trained_models_highlevel/singlePolicy/RobotandHandBlack_DT_2024-05-16-20-42-39/models/model_epoch_epoch=699.ckpt --eval`

Eval real:
`python scripts/evaluation/eval_real.py --config configs/act.json --eval-path /home/rl2-aloha/Documents/EgoplaySP/EgoPlay/trained_models_highlevel/7dimQpos_DT_2024-05-22-14-40-12/7dimQpos_DT_2024-05-22-14-40-12/models/model_epoch_epoch=2949.ckpt`

Use `--debug` to check that the pipeline works


### Single Policy Multi Dataset (Hand + Robot Data)
- `python scripts/pl_train.py --config configs/actSP.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --dataset_2 /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --debug --name pldebug --description debug`


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