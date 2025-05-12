import argparse
import random
from gym.spaces import Box

import numpy as np
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, EnsembleCritic, TanhDiagGaussian
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import EDACPolicy
from rl_preparation.get_rl_data_envs import get_rl_data_envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="edac")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--num-critics", type=int, default=50)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=False)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--normalize-reward", type=bool, default=False)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)

    # for robustness experiments
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--noisy_ratio", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=0.05)

    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="rotation_component1")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cuda_id", type=int, default=0)

    return parser.parse_args()


def train(args=get_args()):
    # offline rl data and env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(args.env, args.task, args.device)

    args.obs_shape = (offline_data['observations'].shape[1], )
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0
    action_space = Box(low=-args.max_action, high=args.max_action, shape=(args.action_dim, ), dtype=np.float32)

    if "betan" in args.task:
        early_stop_epoch_number = 400
    elif "rotation" in args.task:
        early_stop_epoch_number = 500
    else:
        early_stop_epoch_number = 1000

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critics = EnsembleCritic(
        np.prod(args.obs_shape), args.action_dim, \
        args.hidden_dims, num_ensemble=args.num_critics, \
        device=args.device
    )

    # init as in the EDAC paper
    for layer in critics.model[::2]:
        torch.nn.init.constant_(layer.bias, 0.1)
    torch.nn.init.uniform_(critics.model[-1].weight, -3e-3, 3e-3)
    torch.nn.init.uniform_(critics.model[-1].bias, -3e-3, 3e-3)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = EDACPolicy(
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        eta=args.eta
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=offline_data["observations"].shape[0], 
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(offline_data)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["num_critics", "eta"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        noisy_ratio=args.noisy_ratio,
        noise_scale=args.noise_scale,
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        early_stop_epoch_number=early_stop_epoch_number
    )
    
    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()