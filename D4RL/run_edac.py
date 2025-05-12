import argparse
import random
import os, json
import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, EnsembleCritic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import EDACPolicy



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="edac")
    parser.add_argument("--task", type=str, default="walker2d-random-v2")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--num-critics", type=int, default=10)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=False)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--normalize-reward", type=bool, default=False)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stop_epoch_number", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")

    # for robustness experiments
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--noisy_ratio", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=0.05)

    return parser.parse_args()


def train(load_path):

    # set the args
    args = get_args()
    if load_path is not None:
        json_file = load_path + '/hyper_param.json'
        with open(json_file, 'r') as file:
            new_args_dict = json.load(file)
        
        # update the args
        update_terms = ['task', 'seed', 'noise_scale', 'early_stop_epoch_number']
        args_dict = vars(args)
        for k in update_terms:
            if k in new_args_dict:
                args_dict[k] = new_args_dict[k]
            
        args = argparse.Namespace(**args_dict)

    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    if args.normalize_reward:
        mu, std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - mu) / (std + 1e-3)

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # add noise to the data or drop some of the data
    # drop_data(dataset, args.drop_ratio)
    # add_noise_to_data(dataset, args.noisy_ratio, args.noise_scale)

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
            else -np.prod(env.action_space.shape)
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
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

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
    logger.log_hyperparameters(vars(args))

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
        early_stop_epoch_number=args.early_stop_epoch_number
    )
    
    policy_trainer.train()


if __name__ == "__main__":
    current_working_directory = os.getcwd()
    load_path_ls = ['/data/hc-med-exp/seed-0', '/data/hc-med-exp/seed-1', '/data/hc-med-exp/seed-2',
                    '/data/hc-med-rep/seed-0', '/data/hc-med-rep/seed-1', '/data/hc-med-rep/seed-2',
                    '/data/hc-med/seed-0', '/data/hc-med/seed-1', '/data/hc-med/seed-2',
                    '/data/hc-rnd/seed-0', '/data/hc-rnd/seed-1', '/data/hc-rnd/seed-2',
                    '/data/hp-med-exp/seed-0', '/data/hp-med-exp/seed-1', '/data/hp-med-exp/seed-2',
                    '/data/hp-med-rep/seed-0', '/data/hp-med-rep/seed-1', '/data/hp-med-rep/seed-2',
                    '/data/hp-med/seed-0', '/data/hp-med/seed-1', '/data/hp-med/seed-2',
                    '/data/hp-rnd/seed-0', '/data/hp-rnd/seed-1', '/data/hp-rnd/seed-2',
                    '/data/wk-med-exp/seed-0', '/data/wk-med-exp/seed-1', '/data/wk-med-exp/seed-2',
                    '/data/wk-med-rep/seed-0', '/data/wk-med-rep/seed-1', '/data/wk-med-rep/seed-2',
                    '/data/wk-med/seed-0', '/data/wk-med/seed-1', '/data/wk-med/seed-2',
                    '/data/wk-rnd/seed-0', '/data/wk-rnd/seed-1', '/data/wk-rnd/seed-2']
    load_path_id = 35 # 0-6
    train(current_working_directory + load_path_ls[load_path_id])