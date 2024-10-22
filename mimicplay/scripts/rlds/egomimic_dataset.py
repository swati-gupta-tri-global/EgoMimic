import tensorflow_datasets as tfds
import tensorflow as tf
import os
import numpy as np

class EgomimicDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, *args, data_paths=None, **kwargs):
        """
        Args:
            data_paths: Dictionary containing 'train' and 'val' paths for input data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if data_paths:
            self.train_path = os.path.join(data_paths.get('train'))
            self.val_path = os.path.join(data_paths.get('val'))
        else:
            raise ValueError("Input data paths (train and val) must be provided.")

    def _info(self):
        """Returns the dataset metadata."""
        HEIGHT = 480
        WIDTH = 640

        _DESCRIPTION = "EgoMimic data in TFDS/RLDS format"

        observation_features = {
            'front_img_1': tfds.features.Image(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                encoding_format='png',
                doc='Front camera image.'
            ),
            'front_img_1_line': tfds.features.Image(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                encoding_format='png',
                doc='Front camera line image.'
            ),
            'front_img_1_mask': tfds.features.Tensor(
                shape=(HEIGHT, WIDTH),
                dtype=tf.bool,
                doc='Front camera mask image stored as a boolean array.'
            ),
            'front_img_1_masked': tfds.features.Image(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                encoding_format='png',
                doc='Front camera masked image.'
            ),
            'ee_pose': tfds.features.Tensor(
                shape=(None,),  # Variable shape: (3,) or (6,)
                dtype=tf.float64,
                doc='End-effector pose.'
            ),
            'joint_positions': tfds.features.Tensor(
                shape=(None,),  # Variable shape: (7,) or (14,)
                dtype=tf.float32,
                doc='Joint positions of the robot.'
            ),
            'right_wrist_img': tfds.features.Image(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                encoding_format='png',
                doc='Right wrist camera image.'
            ),
            'left_wrist_img': tfds.features.Image(
                shape=(HEIGHT, WIDTH, 3),
                dtype=np.uint8,
                encoding_format='png',
                doc='Left wrist camera image.'
            ),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict(observation_features),
                    'action': tfds.features.Tensor(
                        shape=(None,),  # Variable shape: (3,) or (6,)
                        dtype=tf.float64,
                        doc='Action applied at each time step.'
                    ),
                    'reward': tf.float64,
                    'is_first': tf.bool,
                    'is_last': tf.bool,
                    'is_terminal': tf.bool,
                    'language_prompt': tfds.features.Text(
                        doc='Language prompt associated with the episode.'
                    ),
                }),
                'modality': tfds.features.Text(
                    doc='Modality of the episode.'
                ),
                'episode_metadata': tfds.features.FeaturesDict({
                    'demo_name': tfds.features.Text(
                        doc='Name of the demonstration.'
                    ),
                }),
            }),
            supervised_keys=None,
            homepage='https://egomimic.github.io',
            citation=None,
        )

    def _split_generators(self, dl_manager):
        """Returns splits"""
        if not self.train_path or not self.val_path:
            raise ValueError("Train and validation paths must be provided.")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'path': self.train_path},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={'path': self.val_path},
            ),
        ]
    
    def _generate_examples(self, path):
        """Yields examples"""
        tfrecord_files = tf.io.gfile.glob(os.path.join(path, '*.tfrecord'))
        
        for tfrecord_file in tfrecord_files:
            demo_name = os.path.splitext(os.path.basename(tfrecord_file))[0]
            print(demo_name)

            # Read the TFRecord file as a dataset
            dataset = tf.data.TFRecordDataset(tfrecord_file)

            # Feature description for parsing
            feature_description = {
                # Observation features (serialized tensors)
                'observation/ee_pose': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/front_img_1': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/front_img_1_line': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/front_img_1_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/front_img_1_masked': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/joint_positions': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/right_wrist_img': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'observation/left_wrist_img': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'action': tf.io.FixedLenFeature([], tf.string),
                'language_prompt': tf.io.FixedLenFeature([], tf.string),
                'modality': tf.io.FixedLenFeature([], tf.string),
            }

            # Prepare to collect steps
            steps = []

            # Parse the modality once per episode
            raw_example = next(iter(dataset))
            parsed_example = tf.io.parse_single_example(raw_example, feature_description)
            modality = parsed_example['modality'].numpy().decode('utf-8')
            language_prompt = parsed_example['language_prompt'].numpy().decode('utf-8')

            # Reset dataset iterator
            dataset = tf.data.TFRecordDataset(tfrecord_file)

            # Determine shapes based on modality
            if modality in ['robot', 'hand']:
                ee_pose_shape = (3,)
                action_shape = (3,)
                joint_positions_shape = (7,) if modality == 'robot' else None
            elif modality in ['robot_bimanual', 'hand_bimanual']:
                ee_pose_shape = (6,)
                action_shape = (6,)
                joint_positions_shape = (14,) if modality == 'robot_bimanual' else None
            else:
                raise ValueError(f"Unknown modality: {modality}")

            for idx, raw_record in enumerate(dataset):
                example = tf.io.parse_single_example(raw_record, feature_description)

                observation = {}

                # Parse ee_pose
                ee_pose_raw = example['observation/ee_pose']
                if ee_pose_raw:
                    ee_pose = tf.io.parse_tensor(ee_pose_raw, out_type=tf.float64)
                    ee_pose.set_shape(ee_pose_shape)
                    observation['ee_pose'] = ee_pose.numpy()
                else:
                    assert ee_pose_raw is not None, "EEPOSE must not be None"
                # Parse front_img_1
                front_img_1_raw = example['observation/front_img_1']
                if front_img_1_raw:
                    front_img_1 = tf.io.parse_tensor(front_img_1_raw, out_type=np.uint8)
                    observation['front_img_1'] = front_img_1.numpy()
                else:
                    default = np.zeros([480, 640, 3], dtype=np.uint8)
                    observation['front_img_1'] = default

                # Parse front_img_1_line
                front_img_1_line_raw = example['observation/front_img_1_line']
                if front_img_1_line_raw:
                    front_img_1_line = tf.io.parse_tensor(front_img_1_line_raw, out_type=np.uint8)
                    observation['front_img_1_line'] = front_img_1_line.numpy()
                else:
                    default = np.zeros([480, 640, 3], dtype=np.uint8)
                    observation['front_img_1_line'] = default

                # Parse front_img_1_mask
                front_img_1_mask_raw = example['observation/front_img_1_mask']
                if front_img_1_mask_raw:
                    front_img_1_mask = tf.io.parse_tensor(front_img_1_mask_raw, out_type=tf.bool)
                    observation['front_img_1_mask'] = front_img_1_mask.numpy()
                else:
                    default = np.zeros([480, 640], dtype=np.bool_)
                    observation['front_img_1_mask'] = default

                # Parse front_img_1_masked
                front_img_1_masked_raw = example['observation/front_img_1_masked']
                if front_img_1_masked_raw:
                    front_img_1_masked = tf.io.parse_tensor(front_img_1_masked_raw, out_type=np.uint8)
                    observation['front_img_1_masked'] = front_img_1_masked.numpy()
                else:
                    default = np.zeros([480, 640, 3], dtype=np.uint8)
                    observation['front_img_1_masked'] = default


                # Parse joint_positions (only for robot modalities)
                if modality in ['robot', 'robot_bimanual']:
                    joint_positions_raw = example['observation/joint_positions']
                    if joint_positions_raw:
                        joint_positions = tf.io.parse_tensor(joint_positions_raw, out_type=tf.float32)
                        joint_positions.set_shape(joint_positions_shape)
                        observation['joint_positions'] = joint_positions.numpy()
                    else:
                        observation['joint_positions'] = np.zeros(joint_positions_shape)
                else:
                    observation['joint_positions'] = np.zeros(joint_positions_shape)

                # Parse right_wrist_img (only for robot modalities)
                if modality in ['robot', 'robot_bimanual']:
                    right_wrist_img_raw = example['observation/right_wrist_img']
                    if right_wrist_img_raw:
                        right_wrist_img = tf.io.parse_tensor(right_wrist_img_raw, out_type=np.uint8)
                        observation['right_wrist_img'] = right_wrist_img.numpy()
                    else:
                        # Provide a default image filled with zeros
                        default_image = np.zeros([480, 640, 3], dtype=np.uint8)
                        observation['right_wrist_img'] = default_image
                else:
                    # Provide a default image for modalities without left_wrist_img
                    default_image = np.zeros([480, 640, 3], dtype=np.uint8)
                    observation['right_wrist_img'] = default_image

                # Parse left_wrist_img (only for robot_bimanual modality)
                if modality == 'robot_bimanual':
                    left_wrist_img_raw = example['observation/left_wrist_img']
                    if left_wrist_img_raw:
                        left_wrist_img = tf.io.parse_tensor(left_wrist_img_raw, out_type=np.uint8)
                        observation['left_wrist_img'] = left_wrist_img.numpy()
                    else:
                        # Provide a default image filled with zeros
                        default_image = np.zeros([480, 640, 3], dtype=np.uint8)
                        observation['left_wrist_img'] = default_image
                else:
                    # Provide a default image for modalities without left_wrist_img
                    default_image = np.zeros([480, 640, 3], dtype=np.uint8)
                    observation['left_wrist_img'] = default_image

                # Parse action
                action_raw = example['action']
                action = tf.io.parse_tensor(action_raw, out_type=tf.float64)
                action = action[0].numpy()

                # Parse reward
                reward = 0.0

                is_first = idx == 0

                # Collect the step
                steps.append({
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'is_first': is_first,
                    'is_last': False,
                    'is_terminal': False, # no failure cases
                    'language_prompt': language_prompt,
                })

            if steps:
                steps[-1]['is_last'] = True

            # Yield the episode
            print(f"Yielding {demo_name}")
            yield demo_name, {
                'steps': steps,
                'modality': modality,
                'episode_metadata': {
                    'demo_name': demo_name,
                },
            }

if __name__ == "__main__":
    '''
    python egomimic_dataset.py --dataset /coc/flash7/datasets/egoplay/_OBOO_ROBOTWA/oboo_aug7/converted/rlds/tfds --out /coc/flash7/datasets/egoplay/_OBOO_ROBOTWA/oboo_aug7/converted/rlds/tfds
    '''
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, required=True, help='Directory with the tfrecord files')

    args = parser.parse_args()

    data_paths = {
        'train': os.path.join(args.dataset, 'train'),
        'val': os.path.join(args.dataset, 'val'),
    }

    builder = tfds.builder('egomimic_dataset', data_paths=data_paths)

    builder.download_and_prepare()


