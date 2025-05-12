import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import BayesEnsembleDynamics
from offlinerlkit.utils.scheduler import LinearParameter
from offlinerlkit.buffer import BayesReplayBuffer, SLReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import BAMBRLPolicy
from offlinerlkit.utils.searcher import Searcher
from rl_preparation.get_rl_data_envs import get_rl_data_envs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="bambrl")
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

    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-scheduler", type=bool, default=True)

    # search related
    parser.add_argument("--use-search", type=bool, default=True) # required by ba-mcts and ba-mcts-sl
    parser.add_argument("--search-ratio", type=float, default=0.1) # 0.1
    parser.add_argument("--search-alpha", type=float, default=0.8) # 0.8
    parser.add_argument("--search-ucb-coe", type=float, default=1.0) # 1.0
    parser.add_argument("--search-root-alpha", type=float, default=0.3) # 0.3
    parser.add_argument("--search-n-actions", type=float, default=10) # 10
    parser.add_argument("--search-n-states", type=float, default=5) # 5
    parser.add_argument("--search-n-search", type=float, default=50) # 50

    parser.add_argument("--use-sl", type=bool, default=False) # required by ba-mcts-sl
    parser.add_argument("--sl-policy-only", type=bool, default=True) # only use sl to train the policy (recommended)
    parser.add_argument("--model-retain-epochs-sl", type=int, default=5)
    parser.add_argument("--use-ba", type=bool, default=True)
    parser.add_argument("--sample-step", type=bool, default=False)
    parser.add_argument("--test_search", type=bool, default=True)

    # for robustness experiments
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--noisy_ratio", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=0.05)

    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="rotation_component1") #?
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--search_with_hidden_state", type=bool, default=False) # when you do MCTS, whether to use the hidden state of the rpnn dynamics model; time-costly if true
    parser.add_argument("--cuda_id", type=int, default=1)

    return parser.parse_args()


def train(args=get_args()):    
    if args.use_search:
        args.algo_name += '_mcts'
    if args.use_sl:
        args.algo_name += '_sl'
    
    # offline rl data and env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(args.env, args.task, args.device)

    args.obs_shape = (offline_data['observations'].shape[1], )
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0

    if args.norm_reward:
        r_mean, r_std = offline_data["rewards"].mean(), offline_data["rewards"].std()
        offline_data["rewards"] = (offline_data["rewards"] - r_mean) / (r_std + 1e-3)
    
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
            else -args.action_dim

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        model_path=training_dyn_model_dir,
        device=args.device
    )

    termination_fn = env.is_done
    reward_fn = sa_processor.get_reward
    dynamics = BayesEnsembleDynamics(
        args.sample_step,
        dynamics_model,
        termination_fn,
        reward_fn
    )
    
    # create buffer
    prior_dim = dynamics_model.num_ensemble
    real_buffer = BayesReplayBuffer(
        buffer_size=len(offline_data["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        prior_dim=prior_dim,
        device=args.device
    )
    real_buffer.load_dataset(offline_data, hidden=True)

    fake_buffer = BayesReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        prior_dim=prior_dim,
        device=args.device
    )

    if args.use_sl:
        sl_buffer_size = int(args.rollout_length * args.rollout_batch_size * args.search_ratio * args.model_retain_epochs_sl) # ipt
        sl_buffer = SLReplayBuffer(args.action_dim, np.prod(args.obs_shape), args.search_n_actions, capacity=sl_buffer_size)
        entropy_coe_scheduler = LinearParameter(start=0.01, end=0.001, num_steps=args.epoch * args.step_per_epoch)
    else:
        sl_buffer = None
        entropy_coe_scheduler = None

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=["penalty_coef", "rollout_length", "real_ratio"])
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # create searcher
    if args.use_search:
        searcher = Searcher(args, dynamics, offline_data['state_idxs'], offline_data['action_idxs'], sa_processor)
    else:
        searcher = None

    # create policy
    policy = BAMBRLPolicy(
        args.use_ba,
        args.use_search,
        args.search_ratio,
        args.sl_policy_only,
        searcher,
        sl_buffer,
        entropy_coe_scheduler,
        dynamics,
        actor,
        critics,
        actor_optim,
        critics_optim,
        offline_data['state_idxs'],
        offline_data['action_idxs'],
        sa_processor,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        penalty_coef=args.penalty_coef,
        num_samples=args.num_samples,
        deterministic_backup=args.deterministic_backup,
        max_q_backup=args.max_q_backup
    )

    # get scheduler for the temperature
    # prior_tem_scheduler = LinearScheduler(start_value=100.0, end_value=1.0, num_intervals=1000)
    
    # get Bayes priors associated with the offline dataset
    all_probs = dynamics.get_bayes_priors(offline_data) # (7, 2000000)
    
    temp_prior = 1.0 / dynamics_model.num_ensemble
    uniform_all_prior = np.array([temp_prior for _ in range(dynamics_model.num_ensemble)])
    
    all_priors = [uniform_all_prior]
    trans_num = all_probs.shape[1]
    for i in tqdm(range(trans_num - 1)):
        if args.use_ba:
            done = offline_data['terminals'][i]

            if done:
                all_priors.append(uniform_all_prior)
            else:
                a_prior = all_priors[i]
                a_prob = all_probs[:, i]
                a_prod = a_prior * a_prob
                all_priors.append(a_prod / (a_prod.sum() + 1e-6))
        else:
            all_priors.append(uniform_all_prior)

    all_priors = np.array(all_priors) # (2000000, 7)
    real_buffer.load_prior(all_priors)

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
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
        early_stop_epoch_number=early_stop_epoch_number
    )

    # train policy   
    policy_trainer.train()


if __name__ == "__main__":
    train()