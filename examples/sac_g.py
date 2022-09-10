from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.classic_control import pendulum

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac_g import SACGTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.envs.HM_arena_continuous_task1_max_speed_01_env import HM_arena_continuous_task1_max_speed_01Env

import os
import copy

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


def experiment(variant):
    expl_env = HM_arena_continuous_task1_max_speed_01Env()
    eval_env = HM_arena_continuous_task1_max_speed_01Env()
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    
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
    g_bootstrap = BootstrapTarget(g_target_model, variant["gamma_discount"])
    
    trainer = SACGTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
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




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        gamma_discount=0.99,
        algorithm_kwargs=dict(
            # num_epochs=3000,
            num_epochs=300,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            
        ),
    )
    setup_logger('gamma_test_0', variant=variant)
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    set_device('cpu') # 'cpu' or 'cuda:0' for gamma model device
    experiment(variant)
