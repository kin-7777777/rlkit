from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt

from gamma.td.utils import (
    format_batch_mve,
    soft_update_from_to,
    format_batch_a,
    format_batch_mve
)

SACGLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss g_loss',
)

class SACGTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            g_model,
            g_target_model,
            g_bootstrap,
            
            reward_scale=1.0,
            g_discount=0.99,
            g_sample_discount=None,
            g_mve_discount=0.99,
            g_mve_horizon=3,
            g_tau=0.005,

            policy_lr=1e-3,
            qf_lr=1e-3,
            g_lr=1e-4,
            g_decay=1e-5,
            g_sigma=0.1,
            optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.g_model = g_model
        self.g_target_model = g_target_model
        self.g_bootstrap = g_bootstrap

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.g_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        
        self.g_lr = g_lr
        self.g_decay = g_decay
        self.g_sigma = g_sigma
        
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=self.g_lr, weight_decay=self.g_decay)
        self.g_discount = g_discount
        self.g_sample_discount = g_sample_discount
        if self.g_sample_discount == None:
            self.g_sample_discount = g_discount
        self.g_mve_discount = g_mve_discount
        if self.g_mve_discount <= self.g_discount:
            raise ValueError("SACG Params Error: Discount used for MVE is lower than or equal to gamma model discount!")
        self.g_mve_horizon = g_mve_horizon
        
        self.g_tau = g_tau

        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()
        
        self.g_optimizer.zero_grad()
        losses.g_loss.backward()
        self.g_optimizer.step()
        
        # gamma target model parameters are an exponentially-moving average of model parameters
        soft_update_from_to(self.g_model, self.g_target_model, self.g_tau)

        self._n_train_steps_total += 1

        # self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def sample_gamma_n_rollout(self, batch_size, init_observations, n=3):
        sample_states = init_observations
        for _ in range(n):
            sample_actions = self.policy(sample_states).sample()
            condition_dict = format_batch_mve(sample_states, sample_actions)
            sample_states = self.g_model.sample(batch_size, condition_dict)
        return sample_states

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACGLosses, LossStatistics]:
        from gamma.utils.arrays import DEVICE
        
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        
        batch_size = len(batch['rewards'])

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        
        ## condition dicts contain keys (s, a)
        condition_dict, next_condition_dict = format_batch_a(batch, new_next_actions)
        
        gamma_mve_first_term = 0
        for n in range(1, self.g_mve_horizon+1):
            alpha_n = ((1-self.g_mve_discount) * (self.g_mve_discount-self.g_discount)**(n-1)) / (1-self.g_discount)**n
            rollout_sample_states = self.sample_gamma_n_rollout(batch_size, obs, self.g_mve_horizon).detach()
            rollout_sample_actions = self.policy(rollout_sample_states).sample().detach()
            rollout_sample_rewards = torch.zeros([batch_size, 1]).type(torch.FloatTensor)
            for i in range(batch_size):
                self.env.state = rollout_sample_states[i]
                _, rollout_sample_rewards[i][0], _, _ = self.env.step(ptu.get_numpy(rollout_sample_actions[i]))
            gamma_mve_first_term += alpha_n * rollout_sample_rewards
            
        horizon_sample_states = self.sample_gamma_n_rollout(batch_size, obs, self.g_mve_horizon).detach()
        horizon_sample_actions = self.policy(horizon_sample_states).sample().detach()
        horizon_sample_rewards = torch.zeros([batch_size, 1]).type(torch.FloatTensor)
        for i in range(batch_size):
            self.env.state = horizon_sample_states[i]
            _, horizon_sample_rewards[i][0], _, _ = self.env.step(ptu.get_numpy(horizon_sample_actions[i]))
        gamma_mve_second_term = (1-self.g_mve_discount) * ((self.g_mve_discount-self.g_discount)/(1-self.g_discount))**self.g_mve_horizon * horizon_sample_rewards
        
        v_gamma_mve = gamma_mve_first_term + gamma_mve_second_term
        target_q_values = rewards + self.g_mve_discount * v_gamma_mve.to(torch.device(DEVICE))

        q_target = target_q_values.to(torch.device(DEVICE))
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        Gamma model Loss
        """
        ## update single-step distribution as N(s', σ)
        self.g_bootstrap.update_p(next_condition_dict['s'], sigma=self.g_sigma)
        
        ## sample from bootstrapped target distribution
        samples = self.g_bootstrap.sample(len(rewards),
                                condition_dict, next_condition_dict, discount=self.g_sample_discount)
        
        ## get log-prob of samples under both the target distribution and the model
        log_prob_target = self.g_bootstrap.log_prob(samples, condition_dict, next_condition_dict)
        log_prob_model = self.g_model.log_prob(samples, condition_dict)
        
        ## get g loss
        g_loss = self.g_criterion(log_prob_model, log_prob_target)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics['G Loss'] = np.mean(ptu.get_numpy(g_loss))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACGLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
            g_loss=g_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            # g_model=self.g_model,
            # g_target_model=self.g_target_model,
            # g_bootstrap=self.g_bootstrap,
        )
