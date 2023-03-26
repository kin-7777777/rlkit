import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac_g import SACGTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.networks import Mlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.envs.HM_arena_continuous_task1_max_speed_01_env import HM_arena_continuous_task1_max_speed_01Env
from rlkit.envs.HM_arena_continuous_task1_max_speed_01_env import HM_arena_continuous_task1_simplified

import os
import copy

import numpy as np
import torch
import random

from gamma.flows import (
    make_conditional_flow,
)
from gamma.td.distributions import BootstrapTarget
from gamma.td.structs import (
    ReplayPool,
    Policy,
)
from gamma.utils import (
    mkdir,
    set_device,
)

def sac_g_func(value_disc, model_disc, mve_horizon, seed):

    def experiment(variant):
        expl_env = NormalizedBoxEnv(HM_arena_continuous_task1_simplified())
        eval_env = NormalizedBoxEnv(HM_arena_continuous_task1_simplified())
        obs_dim = expl_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size
        
        ## set seed
        seed = variant['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        eval_env.seed(seed)
        expl_env.seed(seed)
        
        condition_dims = {
            's': obs_dim,
            'a': action_dim,
        }

        M = variant['layer_size']
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        vf = Mlp(
            input_size=obs_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_vf = Mlp(
            input_size=obs_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )
        eval_policy = MakeDeterministic(policy)
        eval_path_collector = MdpPathCollector(
            eval_env,
            eval_policy,
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
        )
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )
        ## initialize conditional spline flow
        g_model = make_conditional_flow(obs_dim, [M, M], condition_dims)
        
        ## target model is analogous to a target Q-function
        g_target_model = copy.deepcopy(g_model)
        
        ## bootstrapped target distribution is mixture of
        ## single-step gaussian (with weight `1 - discount`)
        ## and target model (with weight `discount`)
        if variant['trainer_kwargs']["use_g_mve"]:
            g_bootstrap = BootstrapTarget(g_target_model, variant['trainer_kwargs']["g_mve_discount"])
        else:
            g_bootstrap = BootstrapTarget(g_target_model, variant['trainer_kwargs']["value_discount"])
        
        trainer = SACGTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
            target_vf=target_vf,
            g_model=g_model,
            g_target_model=g_target_model,
            g_bootstrap=g_bootstrap,
            **variant['trainer_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algorithm_kwargs']
        )
        algorithm.to(ptu.device)
        algorithm.train()


    # Run experiment.
    variant = dict(
        algorithm="SACG",
        version="mouse speed 0.01",
        layer_size=256,
        replay_buffer_size=int(2E5),
        seed=seed,
        algorithm_kwargs=dict(
            # num_epochs=300,
            # num_eval_steps_per_epoch=5000,
            # num_trains_per_train_loop=1000,
            # num_expl_steps_per_train_loop=1000,
            # min_num_steps_before_training=1000,
            num_epochs=400,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=250,
            num_expl_steps_per_train_loop=250,
            min_num_steps_before_training=250,
            max_path_length=250,
            batch_size=256,
            vis=False,
            vis_gamma=False,
        ),
        trainer_kwargs=dict(
            policy_lr=1E-4,
            qf_lr=1E-4,
            vf_lr=1E-4,
            soft_target_tau=5e-3,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            value_discount=value_disc,
            g_sample_discount=0.90,
            g_lr=1E-4,
            g_tau = 0.005,
            target_update_period=10,
            g_sigma=0.01,
            use_g_mve=True,
            g_mve_discount=model_disc,
            g_mve_horizon=mve_horizon,
        ),
    )
    
    experiment_name = "sacg_mouse_mve_spd0.01_batch256_"+str(model_disc)
    
    setup_logger(experiment_name, variant=variant)
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    set_device('cpu') # 'cpu' or 'cuda:0' for gamma model device
    experiment(variant)
