import numpy as np
import torch
from typing import NamedTuple
from collections.abc import Generator

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RobustRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_q_values: torch.Tensor
    old_log_dyn_prob: torch.Tensor
    dyn_advantages: torch.Tensor
    dyn_samples: torch.Tensor
    dyn_model_idxs: torch.Tensor

class RolloutBuffer:
    """
    Ref: Stable Baselines 3

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape,
        action_dim,
        device,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size # rollout length
        self.obs_shape = observation_shape
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = device

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class RobustRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape,
        action_dim,
        device,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size,
            observation_shape,
            action_dim,
            device,
            gae_lambda,
            gamma,
            n_envs
        )

    def reset(self) -> None:
        super().reset()
        self.q_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_dyn_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dyn_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dyn_samples = np.zeros((self.buffer_size, self.n_envs, self.obs_shape[0]+1), dtype=np.float32) # can be probelmatic
        self.dyn_model_idxs = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        q_value: torch.Tensor,
        log_dyn_prob: np.ndarray,
        dyn_sample: np.ndarray,
        dyn_model_idx: np.ndarray
    ) -> None:
        super().add(
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob
        )
        # a little bit hacky
        self.q_values[self.pos-1] = q_value.clone().cpu().numpy().flatten()
        self.log_dyn_probs[self.pos-1] = np.array(log_dyn_prob)
        self.dyn_samples[self.pos-1] = np.array(dyn_sample)
        self.dyn_model_idxs[self.pos-1] = np.array(dyn_model_idx)
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, last_q_values: torch.Tensor, dones: np.ndarray) -> None:

        super().compute_returns_and_advantage(last_values, dones)
 
        last_q_values = last_q_values.clone().cpu().numpy().flatten() 
        last_q_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_q_values = last_q_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_q_values = self.q_values[step + 1]
            q_delta = self.rewards[step] + self.gamma * next_q_values * next_non_terminal - self.q_values[step]
            last_q_gae_lam = q_delta + self.gamma * self.gae_lambda * next_non_terminal * last_q_gae_lam
            self.dyn_advantages[step] = last_q_gae_lam
    
    def get_traj(self, batch_size: int):
        assert self.full
        indices = np.random.permutation(self.n_envs)

        start_idx = 0
        while start_idx < self.n_envs:
            yield self._get_sample_trajs(indices[start_idx: (start_idx + batch_size)])
            start_idx += batch_size
    
    def _get_sample_trajs(self, batch_inds: np.ndarray):
        data = (
            self.observations[:, batch_inds],
            self.actions[:, batch_inds],
            self.values[:, batch_inds],
            self.log_probs[:, batch_inds],
            self.advantages[:, batch_inds],
            self.returns[:, batch_inds],
            self.q_values[:, batch_inds],
            self.log_dyn_probs[:, batch_inds],
            self.dyn_advantages[:, batch_inds],
            self.dyn_samples[:, batch_inds],
            self.dyn_model_idxs[:, batch_inds]
        ) # TODO: these are redundant

        return RobustRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    
    def get(self, batch_size: int) -> Generator[RobustRolloutBufferSamples, None, None]:
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "q_values",
                "log_dyn_probs",
                "dyn_advantages",
                "dyn_samples",
                "dyn_model_idxs"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray
    ) -> RobustRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.q_values[batch_inds].flatten(),
            self.log_dyn_probs[batch_inds].flatten(),
            self.dyn_advantages[batch_inds].flatten(),
            self.dyn_samples[batch_inds],
            self.dyn_model_idxs[batch_inds].flatten()
        )
        return RobustRolloutBufferSamples(*tuple(map(self.to_torch, data)))