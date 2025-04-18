"""
HybridCallback Class for Monitoring and Managing Training in Stable-Baselines3.

This callback integrates model saving, evaluation, and logging functionalities for enhanced training monitoring.

Features:
- Saves model checkpoints at regular intervals.
- Evaluates the model on a separate evaluation environment.
- Logs evaluation metrics to TensorBoard and a CSV file.
- Tracks additional rewards-related metrics during rollouts.
- Saves the best-performing model based on evaluation rewards.

Dependencies:
- Stable-Baselines3
- Torch's SummaryWriter for TensorBoard logging
- Custom utility functions: `custom_evaluate_model`, `clear_dir`
- CSV file handling for logging evaluation metrics

Attributes:
    n_eval_episodes (int): Number of episodes for evaluation.
    eval_freq (int): Frequency (in timesteps) for evaluating the model.
    eval_env (gym.Env): The evaluation environment.
    save_freq (int): Frequency (in timesteps) for saving model checkpoints.
    model_dir (str): Directory to save model checkpoints and the best model.
    model_name (str): Base name for saving model checkpoints.
    eval_log_dir (str): Directory for saving evaluation logs (TensorBoard and CSV).
    verbose (int): Verbosity level (0 for silent, >0 for detailed logging).

Methods:
    _init_callback(): Initializes directories for saving the best model.
    _on_step(): Saves model checkpoints, evaluates the model, and logs evaluation metrics.
    on_rollout_end(): Logs episode-level metrics (e.g., goal and collision counts) at the end of a rollout.
    _on_training_end(): Closes resources (e.g., TensorBoard writer) after training ends.
"""

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import os
from params import *
from helper_funcs import *
import csv

class HybridCallback(BaseCallback):
    def __init__(self, n_eval_episodes, eval_freq, eval_env, save_freq, model_dir, model_name, eval_log_dir, verbose=0):
        super(HybridCallback, self).__init__(verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.model_dir = model_dir
        self.model_name = model_name
        self.eval_log_dir = eval_log_dir
        self.verbose = verbose

        self.best_mean_reward = -float('inf')
        self.best_model_dir = f'{model_dir}/best_model'
        self.ep_goalcount = 0
        self.ep_collisioncount = 0
        self.episode_count = 0
        self.writer = SummaryWriter(log_dir=eval_log_dir)

        # Open CSV files for logging
        self.eval_csv_file = open(f'{eval_log_dir}/eval_metrics.csv', mode='w')
        self.eval_csv_writer = csv.writer(self.eval_csv_file)
        self.eval_csv_writer.writerow(['timestep', 'eval_rew_mean', 'eval_goalcount_mean', 'eval_collisioncount_mean'])

    def _init_callback(self) -> None:    
        """
        Initializes the callback by ensuring directories for saving models exist.
        """        
        os.makedirs(self.best_model_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Executes operations at each training step.

        - Saves model checkpoints at regular intervals.
        - Evaluates the model at regular intervals and logs metrics.
        - Tracks and logs episode-level rewards.

        Returns:
            bool: Whether to continue training (always `True`).
        """
        # Save at regular checkpoints
        if self.num_timesteps % self.save_freq == 0:
            ### DAVID's EDIT START ###
            checkpoint_path = os.path.join(self.model_dir, f'{self.model_name}_{self.num_timesteps//1000}k.zip')
            ### DAVID's EDIT END ###
            self.model.save(checkpoint_path)

        # Evaluate at regular checkpoints
        if self.num_timesteps % self.eval_freq == 0:
            eval_metrics = custom_evaluate_model(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, verbose=self.verbose)

            # Log eval metrics to tensorboard
            for metric_name, metric_list in eval_metrics.items():
                self.writer.add_scalar(f'eval/{metric_name}_mean', metric_list[0], self.num_timesteps)

            # Save best model
            if eval_metrics['eval_rew'][0] > self.best_mean_reward:
                self.best_mean_reward = eval_metrics['eval_rew'][0]
                clear_dir(self.best_model_dir)
                self.model.save(f'{self.best_model_dir}/{self.model_name}_{self.num_timesteps//1000}k')
                
                # video_best_model()
                
            # Log to CSV
            self.eval_csv_writer.writerow([self.num_timesteps, eval_metrics['eval_rew'][0], eval_metrics['eval_goalcount'][0], eval_metrics['eval_collisioncount'][0]])

        # LOGGING REWARDS
        infos = self.locals["infos"] # List of info dictionaries for each environment
        dones = self.locals['dones'] # 'dones' array (one flag for each environment)

        # Accumulate episode rewards
        for info, done in zip(infos, dones):
            if done:
                #print(dones)
                self.episode_count += 1 # Increment episode counter for each 'True' flag in 'dones'
            original_env = self.training_env.envs[0].unwrapped
            self.ep_goalcount += info["goal_reward"] // original_env.reward_weights_dict['goal_reward']
            self.ep_collisioncount += info["collision_penalty"] // original_env.reward_weights_dict['obs_collision_penalty_weightage']
        return True  # Continue training
    
    def on_rollout_end(self):
        """
        Logs episode-level metrics at the end of a rollout and resets counters.
        """
        ### ADDITIONAL REWARDS LOGGING ###
        ep_goalcount_mean = self.ep_goalcount / self.episode_count
        ep_collisioncount_mean = self.ep_collisioncount / self.episode_count

        self.logger.record('rollout/ep_goalcount_mean', ep_goalcount_mean)
        self.logger.record('rollout/ep_collisioncount_mean', ep_collisioncount_mean)

        #debugging prints
        if self.verbose > 0:
            print("No of episodes:", self.episode_count)
            print("Total goal count:", self.ep_goalcount)
            print("Total collision count:", self.ep_collisioncount)

        # Reset counters for next rollout
        self.ep_goalcount = 0
        self.ep_collisioncount = 0
        self.episode_count = 0
    
    def _on_training_end(self) -> None:
        """
        Finalizes the callback by closing the TensorBoard writer.
        """
        self.writer.close()  # Close the writer at the end of trainingz