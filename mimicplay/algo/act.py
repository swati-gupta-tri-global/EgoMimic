"""
Implementation of Action Chunking with Transformers (ACT).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils
from mimicplay.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC_VAE
from detr.main import (
    build_ACT_model_and_optimizer,
    build_single_policy_model_and_optimizer,
)
from mimicplay.scripts.aloha_process.simarUtils import nds
import matplotlib.pyplot as plt
import robomimic.utils.obs_utils as ObsUtils

from mimicplay.models.act_nets import Transformer

from robomimic.models.transformers import PositionalEncoding

import robomimic.models.base_nets as BaseNets
import mimicplay.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = ACT, {}

    return algo_class, algo_kwargs


@register_algo_factory_func("actSP")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = ACTSP, {}
    return algo_class, algo_kwargs

class ACTModel(nn.Module):
    '''
    ACT Model closely following DETRVAE from ACT but using standard torch.nn components

    backbones : visual backbone per cam input
    transformer : encoder-decoder transformer
    encoder : style encoder
    latent_dim : style var dim
    a_dim : action dim
    state_dim : proprio dim
    num_queries : predicted action dim
    camera_names : list of camera inputs
    '''
    def __init__(
            self,
            backbones,
            transformer,
            encoder,
            latent_dim,
            a_dim,
            state_dim,
            num_queries,
            camera_names,
    ):
        super(ACTModel, self).__init__()

        self.action_dim = a_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.num_queries = num_queries
        self.camera_names = camera_names

        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d

        self.action_head = nn.Linear(hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(
                10, hidden_dim
            )  # TODO not used in robomimic
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
        '''
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim

        '''
        is_training = actions is not None
        batch_size = qpose.size(0)

        if is_training:
            # Use StyleEncoder to get latent distribution and sample
            dist = self.encoder(qpos, actions)
            mu = dist.mean
            logvar = dist.scale.pow(2).log()
            latent_sample = dist.rsample()
        else:
            # Inference mode, use zeros for latent vector
            mu = logvar = None
            latent_sample = torch.zeros(batch_size, self.latent_dim, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)  # [batch_size, hidden_dim]

        if self.backbones is not None:
            all_cam_features = []
            for cam_id in range(len(self.camera_names)):
                features = self.backbones[cam_id](image[:, cam_id])
                features = self.input_proj(features)
                all_cam_features.append(features)

            src = torch.cat(all_cam_features, dim=-1)  # [B, hidden_dim, H, W * num_cameras]

            batch_size, hidden_dim, height, width = src.shape
            src = src.flatten(2).permute(0, 2, 1)  # [B, S, hidden_dim], S = H * W * num_cameras

            pos_encoding = PositionalEncoding(hidden_dim)
            src = pos_encoding(src.transpose(0, 1)).transpose(0, 1)  # [B, S, hidden_dim]

            proprio_input = self.input_proj_robot_state(qpos).unsqueeze(1)  # [B, 1, hidden_dim]
            latent_input = latent_input.unsqueeze(1)  # [B, 1, hidden_dim]
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]
            tgt = torch.cat([latent_input, proprio_input, query_embed], dim=1)  # [B, 2 + num_queries, hidden_dim]


            # extend tgt
            additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            tgt[:, :2, :] += additional_pos_embed 

            hs = self.transformer(src, tgt) # [B, tgt, hidden_dim]
        else:
            qpos_proj = self.input_proj_robot_state(qpos).unsqueeze(dim=1)  # [B, 1, hidden_dim]
            env_state_proj = self.input_proj_env_state(env_state).unsqueeze(dim=1)  # [B, 1, hidden_dim]
            src = torch.cat([qpos_proj, env_state_proj], dim=1)  # [B, 2, hidden_dim]

            pos_embed = self.pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 2, hidden_dim]

            latent_input = latent_input.unsqueeze(1)  # [B, 1, hidden_dim]
            query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]
            tgt = torch.cat([latent_input, query_embed], dim=1)  # [B, 1 + num_queries, hidden_dim]

            hs = self.transformer(src, tgt, auto_masks=False)
        
        hs_queries = hs[:, 2:, :]
        action_pred = self.action_head(hs_queries)  # [B, num_queries, action_dim]
        is_pad_pred = self.is_pad_head(hs_queries)  # [B, num_queries, 1]

        return action_pred, is_pad_pred, [mu, logvar]

        









class ACT(BC_VAE):
    """
    BC training with a VAE policy.
    """

    def build_model_opt(self, policy_config):
        """
        Builds networks and optimizers for BC algo.
        """
        return build_ACT_model_and_optimizer(policy_config)

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.nets = nn.ModuleDict()
        self.chunk_size = self.global_config["train"]["seq_length"]
        self.camera_keys = self.obs_config["modalities"]["obs"]["rgb"].copy()
        self.proprio_keys = self.obs_config["modalities"]["obs"]["low_dim"].copy()
        self.obs_keys = self.proprio_keys + self.camera_keys

        self.proprio_dim = 0
        for k in self.proprio_keys:
            self.proprio_dim += self.obs_key_shapes[k][0]

        policy_config = {
            "num_queries": self.global_config.train.seq_length,
            "hidden_dim": self.algo_config.act.hidden_dim,
            "dim_feedforward": self.algo_config.act.dim_feedforward,
            "backbone": self.algo_config.act.backbone,
            "enc_layers": self.algo_config.act.enc_layers,
            "dec_layers": self.algo_config.act.dec_layers,
            "nheads": self.algo_config.act.nheads,
            "latent_dim": self.algo_config.act.latent_dim,
            "a_dim": self.ac_dim,
            "ac_key": self.ac_key,
            "state_dim": self.proprio_dim,
            "camera_names": self.camera_keys,
        }
        self.kl_weight = self.algo_config.act.kl_weight
        model, optimizer = self.build_model_opt(policy_config)
        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

        self.temporal_agg = False
        self.query_frequency = self.chunk_size  # TODO maybe tune

        self._step_counter = 0
        self.a_hat_store = None

        rand_kwargs = self.global_config.observation.encoder.rgb.obs_randomizer_kwargs
        self.color_jitter = transforms.ColorJitter(
            brightness=(rand_kwargs.brightness_min, rand_kwargs.brightness_max), contrast=(rand_kwargs.contrast_min, rand_kwargs.contrast_max), saturation=(rand_kwargs.saturation_min, rand_kwargs.saturation_max), hue=(rand_kwargs.hue_min, rand_kwargs.hue_max)
        )

    def process_batch_for_training(self, batch):
        assert False, "Must pass in ac_key for this class"

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
        if ac_key in batch:
            input_batch[ac_key] = batch[ac_key]

        if "type" in batch:
            input_batch["type"] = batch["type"]

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorcal VAEs.
        """

        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _modality_check(self, batch):
        """
        Helper function to check if the batch is robot or hand data.
        """
        if (batch["type"] == 0).all():
            modality = "robot"
        elif (batch["type"] == 1).all():
            modality = "hand"
        else:
            raise ValueError(
                "Got mixed modalities, current implementation expects either robot or hand data only."
            )
        return modality

    def _robomimic_to_act_data(self, batch, cam_keys, proprio_keys):
        proprio = [batch["obs"][k] for k in proprio_keys]
        proprio = torch.cat(proprio, axis=1)
        qpos = proprio

        images = []
        for cam_name in cam_keys:
            image = batch["obs"][cam_name]
            # plt.imsave(f"/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/debug/actAugs/pre{time.time()}.png", image[0].permute(1, 2, 0).cpu().numpy())
            if self.nets.training:
                image = self.color_jitter(image)
            
            # plt.imsave(f"/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/mimicplay/debug/actAugs/post{time.time()}.png", image[0].permute(1, 2, 0).cpu().numpy())
            # image = self.normalize(image)
            image = image.unsqueeze(axis=1)
            images.append(image)
        images = torch.cat(images, axis=1)

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()  # this is not used

        actions = batch[self.ac_key] if self.ac_key in batch else None
        is_pad = batch["obs"]["pad_mask"] == 0  # from 1.0 or 0 to False and True
        is_pad = is_pad.squeeze(dim=-1)
        B, T = is_pad.shape
        assert T == self.chunk_size

        return qpos, images, env_state, actions, is_pad

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

        qpos, images, env_state, actions, is_pad = self._robomimic_to_act_data(
            batch, self.camera_keys, self.proprio_keys
        )

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, actions=actions, is_pad=is_pad
        )
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
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

        qpos, images, env_state, _, is_pad = self._robomimic_to_act_data(
            batch, self.camera_keys, self.proprio_keys
        )
        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, actions=None, is_pad=is_pad
        )

        predictions = OrderedDict()
        predictions[self.ac_key] = a_hat

        if unnorm_stats:
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)

        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        proprio = [obs_dict[k] for k in self.proprio_keys]
        proprio = torch.cat(proprio, axis=1)
        qpos = proprio

        images = []
        for cam_name in self.camera_keys:
            image = obs_dict[cam_name]
            image = self.normalize(image)
            image = image.unsqueeze(axis=1)
            images.append(image)
        images = torch.cat(images, axis=1)

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()  # not used

        if self._step_counter % self.query_frequency == 0:
            a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
                qpos, images, env_state
            )
            self.a_hat_store = a_hat

        action = self.a_hat_store[:, self._step_counter % self.query_frequency, :]
        self._step_counter += 1
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._step_counter = 0

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld


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
