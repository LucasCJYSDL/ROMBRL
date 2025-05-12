import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional

# for PPO
class SDEActor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist,
        log_std_init: float,
        latent_dim_pi: int,
        action_space,
        device
    ) -> None:
        super().__init__()
        self.action_space = action_space
        self.backbone = backbone

        self.dist = dist
        self.dist_net, self.log_std = self.dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=log_std_init
                )
        
        self.device = device

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        mean_actions = self.dist_net(logits)

        return self.dist.proba_distribution(mean_actions, self.log_std, logits)
    
    def reset_noise(self, n_envs):
        self.dist.sample_weights(self.log_std, batch_size=n_envs)
    
    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))


# for SAC/PPO
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist


# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions