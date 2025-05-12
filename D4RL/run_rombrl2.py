import argparse
import random
import json, os
import gym
import d4rl

import numpy as np
import torch

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import ROMBRL2Policy

from offlinerlkit.buffer import RobustRolloutBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="rombrl2")
    parser.add_argument("--task", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--num-q-ensemble", type=int, default=2)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--norm-reward", type=bool, default=False)

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs-since-update", type=int, default=5)
    parser.add_argument("--dynamics-max-epochs", type=int, default=30)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--penalty-coef", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stop_epoch_number", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-scheduler", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    # for robustness experiments
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--noisy_ratio", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=0.05)

    # new hyperparameters
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--onpolicy-rollout-batch-size", type=int, default=250) # important hyperparameter # 250
    parser.add_argument("--onpolicy-rollout-length", type=int, default=100) # 100
    parser.add_argument("--small_traj_batch", type=bool, default=False)
    parser.add_argument("--actor_training_epoch", type=int, default=10)
    parser.add_argument("--actor-dynamics-update-freq", type=int, default=1000)
    parser.add_argument("--onpolicy-batch-size", type=int, default=256)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--grad_mode", type=int, default=1) # corresponding to Eqs. (7) - (9) in the paper

    parser.add_argument("--I_coe", type=float, default=5.0) # TODO: fine-tune
    parser.add_argument("--epsilon", type=float, default=10.0) # TODO: fine-tune
    parser.add_argument("--down_sample_size", type=int, default=8) # m in the paper
    parser.add_argument("--dynamics-adv-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics_training_epoch", type=int, default=10)
    parser.add_argument("--include-ent-in-adv", type=bool, default=False)
    parser.add_argument("--max_inner_epoch", type=int, default=10000)

    # lagrangian-related
    parser.add_argument("--sl_weight", type=float, default=3000.0) # TODO: fine-tune
    parser.add_argument("--lambda_training_epoch", type=int, default=1) # TODO: fine-tune
    parser.add_argument("--lambda_lr", type=float, default=1e-3) # TODO: fine-tune

    return parser.parse_args()


def train(load_path=None):
    args = get_args()
    if load_path is not None:
        json_file = load_path + '/hyper_param.json'
        with open(json_file, 'r') as file:
            new_args_dict = json.load(file)
        
        # update the args
        blocked_terms = ['device', 'algo_name']
        args_dict = vars(args)
        for k, v in new_args_dict.items():
            if k in blocked_terms:
                continue
            if k in args_dict:
                args_dict[k] = v
            
        args = argparse.Namespace(**args_dict)
        args.load_dynamics_path = load_path + '/model'
        # args.load_dynamics_path = None
    
    if args.grad_mode == 3 and args.onpolicy_rollout_batch_size == 2500 and args.small_traj_batch:
        args.small_traj_batch = False

    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)

    if args.norm_reward:
        r_mean, r_std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - r_mean) / (r_std + 1e-3)

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
    critics = []
    for i in range(args.num_q_ensemble):
        critic_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
        critics.append(Critic(critic_backbone, args.device))
    critics = torch.nn.ModuleList(critics)
    critics_optim = torch.optim.Adam(critics.parameters(), lr=args.critic_lr)

    if args.lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    else:
        lr_scheduler = None

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)

    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    ## the buffer for on-policy training
    onpolicy_buffer = RobustRolloutBuffer(args.onpolicy_rollout_length, args.obs_shape, args.action_dim, 
                                          args.device, args.gae_lambda, args.gamma, n_envs=args.onpolicy_rollout_batch_size) # short rollouts with high parallelism

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    dynamics_adv_optim = torch.optim.Adam(
        dynamics_model.parameters(), 
        lr=args.dynamics_adv_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    policy = ROMBRL2Policy(
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup,
        # new
        small_traj_batch=args.small_traj_batch,
        dynamics_adv_optim=dynamics_adv_optim,
        onpolicy_buffer=onpolicy_buffer,
        grad_mode=args.grad_mode,
        I_coe=args.I_coe,
        epsilon=args.epsilon,
        down_sample_size=args.down_sample_size,
        sl_weight=args.sl_weight, 
        lambda_training_epoch=args.lambda_training_epoch,
        lambda_lr=args.lambda_lr,
        onpolicy_rollout_length=args.onpolicy_rollout_length, 
        onpolicy_rollout_batch_size=args.onpolicy_rollout_batch_size,
        onpolicy_batch_size = args.onpolicy_batch_size,
        clip_range = args.clip_range,
        actor_training_epoch = args.actor_training_epoch,
        dynamics_training_epoch = args.dynamics_training_epoch,
        include_ent_in_adv=args.include_ent_in_adv,
        max_inner_epoch=args.max_inner_epoch, # TODO: don't involve this
        device=args.device
    )

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["grad_mode", "noise_scale", "sl_weight", "actor_training_epoch", "onpolicy_rollout_batch_size", "onpolicy_rollout_length", "small_traj_batch"])
    
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
    policy_trainer = MBPolicyTrainer(
        noisy_ratio=args.noisy_ratio,
        noise_scale=args.noise_scale,
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
        dynamics_update_freq=args.actor_dynamics_update_freq, # hacky, but to make full use of the APIs of the original codebase
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
        early_stop_epoch_number=args.early_stop_epoch_number
    )

    # train
    if not load_dynamics_model:
        dynamics.train(
            real_buffer.sample_all(),
            logger,
            max_epochs_since_update=args.max_epochs_since_update,
            max_epochs=args.dynamics_max_epochs
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
    load_path_id = 34 # 0-6
    train(current_working_directory + load_path_ls[load_path_id])