import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

from stable_baselines3.common.buffers import RolloutBuffer


# on-policy trainer, which can only be model-based
class OnPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: RolloutBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int],
        epoch: int,
        batch_size: int,
        eval_setting: Tuple[int, int],
        dynamics_update_freq: int = 0 # for ROMBRL
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_batch_size, self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._batch_size = batch_size
        self._eval_frequency, self._eval_episodes = eval_setting

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        last_10_performance = deque(maxlen=10)
        update_times = 0
        # train loop
        for e in range(1, self._epoch + 1):
            # collect rollouts
            init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
            rollout_info = self.policy.rollout(init_obss, self._rollout_length, self.real_buffer, self.fake_buffer)

            self.logger.log("num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"]))
            for _key, _value in rollout_info.items():
                self.logger.logkv_mean("rollout_info/"+_key, _value)
            
            # train the policy
            update_times += self.policy.learn(self.fake_buffer, self.logger, float(e)/self._epoch)

            # update the dynamics if necessary
            # dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
            # for k, v in dynamics_update_info.items():
            #     self.logger.logkv_mean(k, v)

            if e % self._eval_frequency == 0:
                # evaluate current policy
                eval_info = self._evaluate()
                ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
                ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
                self.logger.logkv("eval/episode_length", ep_length_mean)
                self.logger.logkv("eval/episode_length_std", ep_length_std)
                self.logger.set_timestep(update_times) # TODO: maybe not so reasonable
                self.logger.dumpkvs(exclude=["dynamics_training_progress"])
            
                # save checkpoint
                torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

                # print ETA in hours with epoch number
                elapsed_time = time.time() - start_time
                eta = (elapsed_time / e * (self._epoch - e)) / 3600  # Convert seconds to hours
                print("=" * 20 + f" Epoch {e}/{self._epoch}, ETA: {eta:.2f} hours " + "=" * 20)

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval() # important
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action, _, _, _ = self.policy.select_action(obs.reshape(1, -1), deterministic=True) 
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }