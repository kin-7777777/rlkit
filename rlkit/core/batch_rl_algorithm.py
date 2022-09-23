import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

from rlkit.core import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from gamma.visualization import (
    make_prob_fn,
)
from gamma.utils.arrays import (
    to_torch,
    to_np,
)
from gamma.td.utils import make_condition

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            vis_gamma=False,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self._vis_gamma = vis_gamma

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        eval_paths = self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')
        
        plot_dir_eval = logger._snapshot_dir + "/plots_eval"
        if not os.path.exists(plot_dir_eval):
            os.makedirs(plot_dir_eval)
        plot_dir_expl = logger._snapshot_dir + "/plots_expl"
        if not os.path.exists(plot_dir_expl):
            os.makedirs(plot_dir_expl)
        self.plot_paths(plot_dir_eval, self.epoch, eval_paths)

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sampling', unique=False)
            self.plot_paths(plot_dir_expl, self.epoch, new_expl_paths)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)
            
    # makeshift for mouse environment
    def plot_paths(self, dir, epoch_num, paths, eval=True):
        epoch_dir = dir + "/epoch_" + str(epoch_num)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        reward_halfwidth = 0.4
        for path_idx in range(len(paths)):
            path_x = np.zeros(len(paths[path_idx]['observations']))
            path_y = np.zeros(len(paths[path_idx]['observations']))
            actions = np.zeros(paths[path_idx]['actions'].shape)
            for step_idx in range(len(paths[path_idx]['observations'])):
                path_x[step_idx] = paths[path_idx]['observations'][step_idx][0]
                path_y[step_idx] = paths[path_idx]['observations'][step_idx][1]
                actions[step_idx] = paths[path_idx]['actions'][step_idx]
            plt.scatter(path_x, path_y, c=plt.get_cmap('viridis')(np.linspace(0, 1, len(paths[path_idx]['observations']))))
            plt.plot([-reward_halfwidth, reward_halfwidth], [reward_halfwidth, reward_halfwidth], 'b-')
            plt.plot([-reward_halfwidth, reward_halfwidth], [-reward_halfwidth, -reward_halfwidth], 'b-')
            plt.plot([-reward_halfwidth, -reward_halfwidth], [-reward_halfwidth, reward_halfwidth], 'b-')
            plt.plot([reward_halfwidth, reward_halfwidth], [-reward_halfwidth, reward_halfwidth], 'b-')
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            if paths[path_idx]['dones'][-1]:
                plt.title('reward received')
            plt.savefig(epoch_dir+'/e'+str(epoch_num)+'p'+str(path_idx)+'.png')
            plt.close()
            if self._vis_gamma:
                self.visualize_gamma(dir, epoch_num, path_idx, path_x, path_y, actions, self.trainer.g_model)

    def visualize_gamma(self, dir, epoch_num, path_idx, path_x, path_y, actions, model):
        
        epoch_dir = dir + "/epoch_" + str(epoch_num)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        n_steps = 41
        reward_halfwidth = 0.4
        
        queries, x_range, y_range = self.get_vis_states(n_steps)
        
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        
        steps = np.linspace(0, len(path_x)-1, 4, dtype="int")
        
        for i, ax in enumerate(axes):
            initial_state = np.array([path_x[steps[i]], path_y[steps[i]]])
            initial_action = actions[steps[i]]
        
            cond_rep = initial_state[None].repeat(len(queries), axis=0)
            cond_actions_rep = initial_action[None].repeat(len(queries), axis=0)
            condition_dict = make_condition(cond_rep, cond_actions_rep)
            
            probs = model.log_prob(to_torch(queries), condition_dict)
            probs = to_np(probs.exp())
            probs = probs.reshape(n_steps, n_steps)
            
            handle = ax.imshow(probs, extent=[x_range.min(), x_range.max(),
                    y_range.min(), y_range.max()],
                    aspect='auto')
            
            arrow = np.array([initial_action[0], initial_action[1]])
            arrow_norm = np.linalg.norm(arrow)
            arrow = (arrow / arrow_norm) * 0.3
            ax.scatter(initial_state[0], initial_state[1], color='r', marker='x')
            ax.arrow(initial_state[0], initial_state[1], arrow[0], arrow[1], head_width=0.05, head_length=0.05)
            ax.plot([-reward_halfwidth, reward_halfwidth], [reward_halfwidth, reward_halfwidth], 'b-')
            ax.plot([-reward_halfwidth, reward_halfwidth], [-reward_halfwidth, -reward_halfwidth], 'b-')
            ax.plot([-reward_halfwidth, -reward_halfwidth], [-reward_halfwidth, reward_halfwidth], 'b-')
            ax.plot([reward_halfwidth, reward_halfwidth], [-reward_halfwidth, reward_halfwidth], 'b-')
            
            ax.set_title('actnorm: '+str(arrow_norm), fontsize=8)
            
        plt.suptitle("Total "+str(len(path_x))+" steps")
        plt.savefig(epoch_dir+'/e'+str(epoch_num)+'p'+str(path_idx)+'gamma.png')
        plt.close()
        
    def get_vis_states(self, n_steps, x_range=(-1,1), y_range=(-1,1)):
        x_range = np.linspace(*x_range, n_steps)
        y_range = np.linspace(*y_range, n_steps)

        states = np.array([
            (x, y)
            for x in x_range
            for y in y_range
        ])

        return states, x_range, y_range