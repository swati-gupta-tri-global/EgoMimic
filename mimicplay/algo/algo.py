"""
This file contains base classes that other algorithm classes subclass.
Each algorithm file also implements a algorithm factory function that
takes in an algorithm config (`config.algo`) and returns the particular
Algo subclass that should be instantiated, along with any extra kwargs.
These factory functions are registered into a global dictionary with the
@register_algo_factory_func function decorator. This makes it easy for
@algo_factory to instantiate the correct `Algo` subclass.
"""

import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch.nn as nn

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils


# mapping from algo name to factory functions that map algo configs to algo class names
REGISTERED_ALGO_FACTORY_FUNCS = OrderedDict()


def register_algo_factory_func(algo_name):
    """
    Function decorator to register algo factory functions that map algo configs to algo class names.
    Each algorithm implements such a function, and decorates it with this decorator.

    Args:
        algo_name (str): the algorithm name to register the algorithm under
    """

    def decorator(factory_func):
        REGISTERED_ALGO_FACTORY_FUNCS[algo_name] = factory_func

    return decorator


def algo_name_to_factory_func(algo_name):
    """
    Uses registry to retrieve algo factory function from algo name.

    Args:
        algo_name (str): the algorithm name
    """
    return REGISTERED_ALGO_FACTORY_FUNCS[algo_name]


def algo_factory(algo_name, config, obs_key_shapes, ac_dim, device):
    """
    Factory function for creating algorithms based on the algorithm name and config.

    Args:
        algo_name (str): the algorithm name

        config (BaseConfig instance): config object

        obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

        ac_dim (int): dimension of action space

        device (torch.Device): where the algo should live (i.e. cpu, gpu)
    """

    # @algo_name is included as an arg to be explicit, but make sure it matches the config
    assert algo_name == config.algo_name

    # use algo factory func to get algo class and kwargs from algo config
    factory_func = algo_name_to_factory_func(algo_name)
    algo_cls, algo_kwargs = factory_func(config.algo)

    # create algo instance
    return algo_cls(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
        **algo_kwargs
    )


class Algo(object):
    """
    Base algorithm class that all other algorithms subclass. Defines several
    functions that should be overriden by subclasses, in order to provide
    a standard API to be used by training functions such as @run_epoch in
    utils/train_utils.py.
    """

    def __init__(
        self, algo_config, obs_config, global_config, obs_key_shapes, ac_dim, device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.nets = nn.ModuleDict()
        self._create_shapes(obs_config.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _create_shapes(self, obs_keys, obs_key_shapes):
        """
        Create obs_shapes, goal_shapes, and subgoal_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. Each dictionary
        maps observation key to shape.

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"obs": ["rgb", "proprio"], "goal": ["proprio"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
        """
        # determine shapes
        self.obs_shapes = OrderedDict()
        self.goal_shapes = OrderedDict()
        self.subgoal_shapes = OrderedDict()

        # We check across all modality groups (obs, goal, subgoal), and see if the inputted observation key exists
        # across all modalitie specified in the config. If so, we store its corresponding shape internally
        for k in obs_key_shapes:
            if "obs" in self.obs_config.modalities and k in [
                obs_key
                for modality in self.obs_config.modalities.obs.values()
                for obs_key in modality
            ]:
                self.obs_shapes[k] = obs_key_shapes[k]
            if "goal" in self.obs_config.modalities and k in [
                obs_key
                for modality in self.obs_config.modalities.goal.values()
                for obs_key in modality
            ]:
                self.goal_shapes[k] = obs_key_shapes[k]
            if "subgoal" in self.obs_config.modalities and k in [
                obs_key
                for modality in self.obs_config.modalities.subgoal.values()
                for obs_key in modality
            ]:
                self.subgoal_shapes[k] = obs_key_shapes[k]

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        @self.nets should be a ModuleDict.
        """
        raise NotImplementedError

    def _create_optimizers(self):
        """
        Creates optimizers using @self.optim_params and places them into @self.optimizers.
        """
        self.optimizers = dict()
        self.lr_schedulers = dict()

        for k in self.optim_params:
            # only make optimizers for networks that have been created - @optim_params may have more
            # settings for unused networks
            if k in self.nets:
                if isinstance(self.nets[k], nn.ModuleList):
                    self.optimizers[k] = [
                        TorchUtils.optimizer_from_optim_params(
                            net_optim_params=self.optim_params[k], net=self.nets[k][i]
                        )
                        for i in range(len(self.nets[k]))
                    ]
                    self.lr_schedulers[k] = [
                        TorchUtils.lr_scheduler_from_optim_params(
                            net_optim_params=self.optim_params[k],
                            net=self.nets[k][i],
                            optimizer=self.optimizers[k][i],
                        )
                        for i in range(len(self.nets[k]))
                    ]
                else:
                    self.optimizers[k] = TorchUtils.optimizer_from_optim_params(
                        net_optim_params=self.optim_params[k], net=self.nets[k]
                    )
                    self.lr_schedulers[k] = TorchUtils.lr_scheduler_from_optim_params(
                        net_optim_params=self.optim_params[k],
                        net=self.nets[k],
                        optimizer=self.optimizers[k],
                    )

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        return batch

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        assert validate or self.nets.training
        return OrderedDict()

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss log (dict): name -> summary statistic
        """
        log = OrderedDict()

        # record current optimizer learning rates
        for k in self.optimizers:
            for i, param_group in enumerate(self.optimizers[k].param_groups):
                log["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]

        return log

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch.
        """

        # LR scheduling updates
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()

    def set_eval(self):
        """
        Prepare networks for evaluation.
        """
        self.nets.eval()

    def set_train(self):
        """
        Prepare networks for training.
        """
        self.nets.train()

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return self.nets.state_dict()

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict)

    def __repr__(self):
        """
        Pretty print algorithm and network description.
        """
        return (
            "{} (\n".format(self.__class__.__name__)
            + textwrap.indent(self.nets.__repr__(), "  ")
            + "\n)"
        )

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        pass


class PolicyAlgo(Algo):
    """
    Base class for all algorithms that can be used as policies.
    """

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError


class ValueAlgo(Algo):
    """
    Base class for all algorithms that can learn a value function.
    """

    def get_state_value(self, obs_dict, goal_dict=None):
        """
        Get state value outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError

    def get_state_action_value(self, obs_dict, actions, goal_dict=None):
        """
        Get state-action value outputs.

        Args:
            obs_dict (dict): current observation
            actions (torch.Tensor): action
            goal_dict (dict): (optional) goal

        Returns:
            value (torch.Tensor): value tensor
        """
        raise NotImplementedError


class PlannerAlgo(Algo):
    """
    Base class for all algorithms that can be used for planning subgoals
    conditioned on current observations and potential goal observations.
    """

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get predicted subgoal outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal prediction (dict): name -> Tensor [batch_size, ...]
        """
        raise NotImplementedError

    def sample_subgoals(self, obs_dict, goal_dict, num_samples=1):
        """
        For planners that rely on sampling subgoals.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoals (dict): name -> Tensor [batch_size, num_samples, ...]
        """
        raise NotImplementedError


class HierarchicalAlgo(Algo):
    """
    Base class for all hierarchical algorithms that consist of (1) subgoal planning
    and (2) subgoal-conditioned policy learning.
    """

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        raise NotImplementedError

    def get_subgoal_predictions(self, obs_dict, goal_dict=None):
        """
        Get subgoal predictions from high-level subgoal planner.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            subgoal (dict): predicted subgoal
        """
        raise NotImplementedError

    @property
    def current_subgoal(self):
        """
        Get the current subgoal for conditioning the low-level policy

        Returns:
            current subgoal (dict): predicted subgoal
        """
        raise NotImplementedError


class RolloutPolicy(object):
    """
    Wraps @Algo object to make it easy to run policies in a rollout loop.
    """

    def __init__(self, policy, obs_normalization_stats=None):
        """
        Args:
            policy (Algo instance): @Algo object to wrap to prepare for rollouts

            obs_normalization_stats (dict): optionally pass a dictionary for observation
                normalization. This should map observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        self.policy = policy
        self.obs_normalization_stats = obs_normalization_stats

    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.policy.set_eval()
        self.policy.reset()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
        """
        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        if self.obs_normalization_stats is not None:
            # ensure obs_normalization_stats are torch Tensors on proper device
            obs_normalization_stats = TensorUtils.to_float(
                TensorUtils.to_device(
                    TensorUtils.to_tensor(self.obs_normalization_stats),
                    self.policy.device,
                )
            )
            # limit normalization to obs keys being used, in case environment includes extra keys
            ob = {k: ob[k] for k in self.policy.global_config.all_obs_keys}
            ob = ObsUtils.normalize_batch(
                ob, obs_normalization_stats=obs_normalization_stats
            )
        return ob

    def __repr__(self):
        """Pretty print network description"""
        return self.policy.__repr__()

    def __call__(self, ob, goal=None):
        """
        Produce action from raw observation dict (and maybe goal dict) from environment.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
            goal (dict): goal observation
        """
        ob = self._prepare_observation(ob)
        if goal is not None:
            goal = self._prepare_observation(goal)
        ac = self.policy.get_action(obs_dict=ob, goal_dict=goal)
        return TensorUtils.to_numpy(ac[0])


class ACTSP(ACT):
    def _create_networks(self):
        super(ACTSP, self)._create_networks()
        self.proprio_keys_hand = (
            self.global_config.observation_hand.modalities.obs.low_dim.copy()
        )

        self.ac_key_hand = self.global_config.train.ac_key_hand
        self.ac_key_robot = self.global_config.train.ac_key

        # self.proprio_dim = 0
        # for k in self.proprio_keys_hand:
        #     self.proprio_dim_hand += self.obs_key_shapes[k][0]

    def build_model_opt(self, policy_config):
        return build_single_policy_model_and_optimizer(policy_config)
    
    def process_batch_for_training(self, batch, ac_key):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        input_batch = dict()
        input_batch["obs"] = {
            k: batch["obs"][k][:, 0, :]
            for k in batch["obs"]
            if k != "pad_mask" and k != "type"
        }
        input_batch["obs"]["pad_mask"] = batch["obs"]["pad_mask"]
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present

        if self.ac_key_hand in batch:
            input_batch[self.ac_key_hand] = batch[self.ac_key_hand]
        if self.ac_key_robot in batch:
            input_batch[self.ac_key_robot] = batch[self.ac_key_robot]

        if "type" in batch:
            input_batch["type"] = batch["type"]

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _robomimic_to_act_data(self, batch, cam_keys, proprio_keys):
        qpos, images, env_state, actions, is_pad = super()._robomimic_to_act_data(batch, cam_keys, proprio_keys)
        actions_hand = batch.get(self.ac_key_hand, None)
        actions_robot = batch[self.ac_key_robot] if self.ac_key_robot in batch else None

        return qpos, images, env_state, actions_hand, actions_robot, is_pad

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        modality = self._modality_check(batch)
        cam_keys = (
            self.camera_keys if modality == "robot" else self.camera_keys[:1]
        )  # TODO Simar rm hardcoding
        proprio_keys = (
            self.proprio_keys_hand if modality == "hand" else self.proprio_keys
        )
        qpos, images, env_state, actions_hand, actions_robot, is_pad = self._robomimic_to_act_data(
            batch, cam_keys, proprio_keys
        )
    
        actions = actions_hand if modality == "hand" else actions_robot

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, modality, actions=actions, is_pad=is_pad
        )
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict = dict()

        if modality == "hand":
            all_l1 = F.l1_loss(actions_hand, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean() * self.global_config.algo.sp.hand_lambda
            total_kld = total_kld * self.global_config.algo.sp.hand_lambda
        elif modality == "robot":
            all_l1_robot = F.l1_loss(actions_robot, a_hat[0], reduction="none")
            all_l1_hand = F.l1_loss(actions_hand, a_hat[1], reduction="none")
            l1 = (all_l1_robot * ~is_pad.unsqueeze(-1)).mean() + (all_l1_hand * ~is_pad.unsqueeze(-1)).mean()

        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict["kl"],
            reconstruction_loss=loss_dict["l1"],
        )

        return predictions

    def forward_eval(self, batch, unnorm_stats):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        modality = self._modality_check(batch)

        cam_keys = (
            self.camera_keys if modality == "robot" else self.camera_keys[:1]
        )  # TODO Simar rm hardcoding
        proprio_keys = (
            self.proprio_keys_hand if modality == "hand" else self.proprio_keys
        )
        qpos, images, env_state, _, _, is_pad = self._robomimic_to_act_data(
            batch, cam_keys, proprio_keys
        )
        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, modality, actions=None, is_pad=is_pad
        )

        # a_hat = a_hat[0] if modality == "robot" else a_hat
        if modality == "robot":
            predictions = OrderedDict()
            predictions[self.ac_key_robot] = a_hat[0]
            predictions[self.ac_key_hand] = a_hat[1]
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)
        else:
            predictions = OrderedDict()
            predictions[self.ac_key_hand] = a_hat
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)

        return predictions

class TestModel:
    def __init__(self, config_path):
        ext_cfg = self.load_config(config_path)
        config = config_factory(ext_cfg["algo_name"])

        with config.values_unlocked():
            config.update(ext_cfg)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ac_dim = 7  # Action dimension
        obs_key_shapes = {
            'joint_positions': (7,),
            'front_img_1': (3, 480, 640),
            'right_wrist_img': (3, 480, 640),
        }

        self.act_algo = ACT(
            algo_config=config.algo,  # Use the appropriate section from the config object
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device,
        )

        self.act_algo.train_config = config.train

        # Create networks
        self.act_algo._create_networks()

    def load_config(self, config_path):
        """Load the config from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def run_test(self):
        batch_size = 2
        seq_length = self.act_algo.train_config.seq_length  # Access via the config object

        # Create a dummy batch
        dummy_batch = {
            'obs': {
                'joint_positions': torch.randn(batch_size, seq_length, *self.act_algo.obs_key_shapes['joint_positions']),
                'front_img_1': torch.randint(0, 256, (batch_size, seq_length, *self.act_algo.obs_key_shapes['front_img_1']), dtype=torch.uint8),
                'right_wrist_img': torch.randint(0, 256, (batch_size, seq_length, *self.act_algo.obs_key_shapes['right_wrist_img']), dtype=torch.uint8),
                'pad_mask': torch.ones(batch_size, seq_length, 1),
            },
            'actions_joints_act': torch.randn(batch_size, seq_length, self.act_algo.ac_dim),
        }

        # Process the batch for training
        batch = self.act_algo.process_batch_for_training(dummy_batch, 'actions_joints_act')

        print("Processed Batch:", batch)

        # Move batch to device
        batch = self.to_device(batch, self.act_algo.device)

        # Ensure the model is in training mode
        self.act_algo.nets['policy'].train()

        # Perform forward pass for training
        predictions = self.act_algo._forward_training(batch)

        # Compute losses
        losses = self.act_algo._compute_losses(predictions, batch)

        print("Predictions:")
        for key, value in predictions.items():
            print(f"{key}: {value}")

        print("\nLosses:")
        for key, value in losses.items():
            print(f"{key}: {value.item() if isinstance(value, torch.Tensor) else value}")

    def to_device(self, batch, device):
        """Utility function to move data to the specified device."""
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v, device) for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        else:
            return batch
