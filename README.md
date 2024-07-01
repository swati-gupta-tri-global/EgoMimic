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
python aloha_to_robomimicv2.py --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/rawAloha --arm right --out /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5  --extrinsics humanoidApr16 --data-type robot
```


### Calibration
- cd into `mimicplay/scripts/calibrate_camera`
- Run `python calibrate_egoplay.py --h5py-path <path to hdf5 from previous section.hdf5>`
- This will output the transform matrices


### Hand Data
- Run `hand_data_concat.py`
- Run `EgoPlay/mimicplay/scripts/aloha_process/mimicplay_data_process.py`



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
- `python scripts/pl_train.py --config configs/actSP.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --dataset_2 /coc/flash7/datasets/egoplay/oboo_diverse_aria_may9/converted/aria_oboo_diverseMimicplay.hdf5 --debug --name pldebug --description debug`

Yellow table only
- `python scripts/pl_train.py --config configs/actSP.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --dataset_2 /coc/flash7/datasets/egoplay/one_bowl_one_object/plushiesMimicplay_with_type_label.hdf5 --debug --name pldebug --description debug`

Yellow + black table
- `python scripts/pl_train.py --config configs/actSP.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --dataset_2 /coc/flash7/datasets/egoplay/one_bowl_one_object/plushiesMimicplay_hand_yellow_black_table_with_type_label.hdf5 --debug --name pldebug --description debug`

Masking
- `python scripts/pl_submit.py --config configs/actSP.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/maskedRobot.hdf5 --dataset_2 /coc/flash7/datasets/egoplay/one_bowl_one_object_masked/plushiesLineMaskedMimicplay_label.hdf5 --name singlePolicy --description RobotandHand --num-nodes 1 --gpus-per-node 4 --batch-size 32 --lr 5e-5`


## Training policies (Without PL)
Base High level:
`python scripts/train.py --config configs/highlevel_real.json --dataset /coc/flash7/datasets/egoplay/humanoidStacking/humanoid_stackingMimicplay.hdf5 --name humanoidStacking --description v1`
or via `python scripts/exps/submit.py`

With dinov2 non goal cond
`python scripts/train.py --config configs/highlevel_dino_lora.json --dataset /coc/flash7/datasets/egoplay/oboo_depth_apr22/oboo_robot_apr22_Mimicplay.hdf5 --name oboo --description vanillaRobot --non-goal-cond`

### Training on multiple datasets
```bash
python scripts/train.py --config configs/highlevel_dino_2_train_datasets.json --dataset /coc/flash7/datasets/egoplay/one_bowl_one_object/plushiesMimicplay_hand_yellow_black_table_with_type_label.hdf5 --dataset_2 /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --name debug --description debug --non-goal-cond --debug
```

- #### Co-train
Set `cotrain=true` in `../configs/highlevel_dino_2_train_datasets.json`

- #### Co-train+KL 
Set `cotrain=true` and `KL=true` in `../configs/highlevel_dino_2_train_datasets.json`

- #### Co-train + Domain Discriminator
Set `cotrain=true` and `domain_discriminator=true` in  `../configs/highlevel_dino_2_train_datasets.json`

### Training on single dataset
```bash
python train.py --config ../configs/highlevel_dino_2_train_datasets.json --dataset <path-to-dataset> --name <exp-name> --description no_goal --non-goal-cond
```

Set `co-train`, `kl`, `domain_discriminator` as `false` in  `../configs/highlevel_dino_2_train_datasets.json`

## Update Notes
Dirty laundry
- Color jitter is manually implemented for ACT in _robomimic_to_act_data rather than using the ObsUtils color jitter
- hardcoded extrinsics in val_utils.py
- Added ac_key under base Algo in robomimic, I suppose this could just access the model.global_config
- I haven't tested whether aloha_to_robomimic_v2 works with highlevelGMMPretrain

- Remember: type == 0 is robot, type==1 is hand
- Now that our hdf5's have either hand or robot data, we can just specify this from the config.  So simply set config.train.data_type and config.train.data2_type to hand or robot.


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