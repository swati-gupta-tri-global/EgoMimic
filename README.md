# Egoplay
## Installation
Create and activate conda environment
```	
conda create -n mimicplay python=3.8
conda activate mimicplay
```

MimicPlay is based on [robomimic](https://github.com/ARISE-Initiative/robomimic), which facilitates the basics of learning from offline demonstrations.
```	
cd ..
git clone https://github.com/SimarKareer/robomimic
cd robomimic
git checkout <custom branch>
pip install -e .
cd act/detr
pip install -e .
```

Install EgoPlay
```	
cd ..
git clone https://github.com/SimarKareer/EgoPlay/tree/main
cd EgoPlay
pip install -e .
```

Install other things
```
pip install git+https://github.com/simarkareer/submitit
pip install av
pip install pynvml
```

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
python aloha_to_robomimicv2.py --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/rawAloha --arm right --out /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5  --extrinsics humanoidApr16
```


### Calibration
- cd into `mimicplay/scripts/calibrate_camera`
- Run `python calibrate_egoplay.py --h5py-path <path to hdf5 from previous section.hdf5>`
- This will output the transform matrices


### Hand Data
- Run `hand_data_concat.py`
- Run `EgoPlay/mimicplay/scripts/aloha_process/mimicplay_data_process.py`


## Training policies
Base High level:
`python scripts/train.py --config configs/highlevel_real.json --dataset /coc/flash7/datasets/egoplay/humanoidStacking/humanoid_stackingMimicplay.hdf5 --name humanoidStacking --description v1`
or via `python scripts/exps/submit.py`

With dinov2 non goal cond
`python scripts/train.py --config configs/highlevel_dino_lora.json --dataset /coc/flash7/datasets/egoplay/one_bowl_one_object_robot_apr9/robomimic/oboo_apr9Mimicplay.hdf5 --name oboo --description vanillaRobot --non-goal-cond`

With ACT settings
`python scripts/submit.py --config configs/act.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --name vanillaACT --description joints --non-goal-cond --ac-key actions_joints --obs-rgb front_img_1 right_wrist_img`

Debugging pl
`python scripts/pl_train.py --config configs/act.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --debug --name pldebug --description debug`

Launching with pl
`python scripts/pl_submit.py --config configs/act.json --dataset /coc/flash7/datasets/egoplay/oboov2_robot_apr16/oboov2_robot_apr16ACT.hdf5 --name vanillaACTPL --description 8GBS32LR5e5 --num-nodes 1 --gpus-per-node 8 --batch-size 32 --lr 1e-4`

Use `--debug` to check that the pipeline works

Remember `conda activate eplay2`

Dirty laundry
- Hardcoded path to urdf in SimarUtils.py
- hardcoded extrinsics in val_utils.py

Eval real:
`python scripts/evaluation/eval_real.py --config configs/act.json --eval-path /home/rl2-aloha/Documents/EgoplaySP/EgoPlay/trained_models_highlevel/1GBS32LR5e5_DT_2024-05-01-11-47-59/1GBS32LR5e5_DT_2024-05-01-11-47-59/models/model_epoch_epoch=599.ckpt`