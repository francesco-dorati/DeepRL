import argparse
import random
import numpy as np 
import torch
import os
import gymnasium as gym
from setuptools._distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument('--gym-id', type=str, default="CartPole-v1", help="the id of the gym environment")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="the learning rate")
    parser.add_argument('--seed', type=int, default=1, help="the seed")
    parser.add_argument('--total-timesteps', type=int, default=25000, help="total timesteps")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="torch deterministic")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="cuda enabled")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help="tracked by w&b")
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL", help="w&b project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="w&b entity name")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help="w&b entity name")

    # Algorithm specific
    parser.add_argument('--num-envs', type=int, default=4, help="total timesteps")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writer.add_scalar("test_loss", i*2, global_step=i)

    # DO NOT MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(gym_id, render_mode="rgb_array", seed=seed)
            env = gym.wrappers.RecordEpisodeStatistics(env) # for episodic return
            if capture_video:
                if idx == 0:
                    video_folder = f"videos/{run_name}"
                    env = gym.wrappers.RecordVideo(
                        env,
                        video_folder=video_folder,
                        episode_trigger=lambda episode_id: episode_id % 100 == 0
                    )
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    # envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    # try:
    #     observation = envs.reset()
    #     for _ in range(200):
    #         action = envs.action_space.sample()
    #         observation, reward, terminated, truncated, infos = envs.step(action)
    #         # infos is a list, one per env
    #         for i, info in enumerate(infos):
    #             # Only print if it's a dict and has 'episode'
    #             if isinstance(info, dict) and "episode" in info:
    #                 print(f"episodic return (env {i}): {info['episode']['r']}")
    # finally:
    #     for env in envs.envs:
    #         try:
    #             env.close()
    #         except Exception:
    #             pass

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete actions are supported"

    print("Observation space shape", envs.single_observation_space.shape)
    print("Action space shape", envs.single_action_space.shape)


