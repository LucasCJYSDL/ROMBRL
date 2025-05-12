import numpy as np
import torch
from torch.nn import functional as F

from typing import Dict
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.policy import BasePolicy
from offlinerlkit.buffer import ReplayBuffer

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance


class PPOPolicy(BasePolicy):
    """
    Ref: Stable Baselines 3
    """

    def __init__(
        self,
        lr_schedule,
        batch_size,
        clip_range,
        clip_range_vf, 
        ent_coef,
        vf_coef,
        target_kl,
        max_grad_norm,
        use_sde: bool,
        inner_epoch: int,
        dynamics: BaseDynamics,
        agent: MlpPolicy,
        device,
        scaler = None       
    ) -> None:
        super().__init__()

        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.inner_epoch = inner_epoch

        self.dynamics = dynamics
        self.agent = agent
        self.device = device
        self.scaler = scaler
    
    def train(self) -> None:
        self.agent.set_training_mode(True)
    
    def eval(self) -> None:
        self.agent.set_training_mode(False)

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int,
        real_buffer: ReplayBuffer,
        rollout_buffer: RolloutBuffer
    ) -> Dict:
        
        self.eval()
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.agent.reset_noise(init_obss.shape[0]) # TODO: reset it during rollouts as well

        # rollout information
        num_transitions = 0
        rewards_arr = np.array([])

        # rollout
        observations = init_obss
        last_terminals = np.zeros((init_obss.shape[0], ), dtype=bool)

        for step_id in range(rollout_length):
            step_actions, actions, values, log_probs = self.select_action(observations, deterministic=False)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, step_actions)

            rewards = rewards.flatten()
            terminals = terminals.flatten()
            num_terminals = terminals.sum()
            if num_terminals > 0:
                new_observations = real_buffer.sample(num_terminals)["observations"].cpu().numpy()
                next_observations[terminals] = new_observations
            
            rollout_buffer.add(observations, actions, rewards, last_terminals, values, log_probs)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards)

            observations = next_observations
            last_terminals = terminals
        
        # after rollout
        with torch.no_grad():
            # Compute value for the last timestep

            if self.scaler is not None:
                next_observations = self.scaler.transform(next_observations)

            values = self.agent.predict_values(torch.FloatTensor(next_observations).to(self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=terminals)

        return {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def select_action(
        self,
        obs: np.ndarray,
        deterministic
    ):
        if self.scaler is not None:
            obs = self.scaler.transform(obs)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            actions, values, log_probs = self.actforward(obs_tensor, deterministic)

        actions = actions.cpu().numpy() # this action has been squashed
        step_actions = self.agent.unscale_action(actions)

        return step_actions, actions, values, log_probs

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic
    ):
        actions, values, log_probs = self.agent(obs, deterministic)

        return actions, values, log_probs

    def learn(self, rollout_buffer: RolloutBuffer, logger, training_progress):
        # switch to train mode (this affects batch norm / dropout)
        self.train() 

        # update optimizer learning rate, TODO: update the clip range based on the training progress as well
        learning_rate = self.lr_schedule(1.0 - training_progress)
        for param_group in self.agent.optimizer.param_groups:
            param_group["lr"] = learning_rate

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        gd_times = 0
        # train for n_epochs epochs
        for epoch in range(self.inner_epoch):
            approx_kl_divs = []

            # do a complete pass on the rollout buffer?
            batch_num = 0
            for rollout_data in rollout_buffer.get(self.batch_size): # TODO: maybe too much
                actions = rollout_data.actions

                if self.scaler is not None:
                    obses = self.scaler.transform_tensor(rollout_data.observations)
                else:
                    obses = rollout_data.observations

                values, log_prob, entropy = self.agent.evaluate_actions(obses, actions)
                values = values.flatten()

                # normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # no clipping
                    values_pred = values
                else:
                    # clip the difference between old and new value; NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
                    )

                # value loss using the TD (gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # entropy loss favor exploration
                if entropy is None:
                    # approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")

                # Optimization step
                self.agent.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()

                gd_times += 1 # number of gradient descent steps
                batch_num += 1

                if batch_num > 80: # danger
                    break

            if not continue_training:
                break
        
        explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

        # Logs
        logger.logkv("train/entropy_loss", np.mean(entropy_losses))
        logger.logkv("train/policy_gradient_loss", np.mean(pg_losses))
        logger.logkv("train/value_loss", np.mean(value_losses))
        logger.logkv("train/approx_kl", np.mean(approx_kl_divs))
        logger.logkv("train/clip_fraction", np.mean(clip_fractions))
        logger.logkv("train/loss", loss.item())
        logger.logkv("train/explained_variance", explained_var)

        return gd_times