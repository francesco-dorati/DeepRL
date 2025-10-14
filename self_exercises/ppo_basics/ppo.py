import os
import time
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# Parameters
# -----------------------------
ENV_ID = "CartPole-v1"          # Gymnasium environment
TOTAL_TIMESTEPS = 500_000        # Training steps
NUM_ENVS = 4                    # Vectorized environments
VIDEO_FREQ = 5000               # Record video every n steps
VIDEO_FOLDER = "videos"
RUN_NAME = f"{ENV_ID}_ppo_{int(time.time())}"
SEED = 42
LOG_DIR = f"runs/{RUN_NAME}"

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
# TensorBoard writer
# -----------------------------
writer = SummaryWriter(LOG_DIR)

# -----------------------------
# Custom callback for logging
# -----------------------------
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if "episode" in self.locals['infos'][0]:
            ep_info = self.locals['infos'][0]['episode']
            writer.add_scalar("charts/episode_reward", ep_info['r'], self.num_timesteps)
            writer.add_scalar("charts/episode_length", ep_info['l'], self.num_timesteps)
        return True

# -----------------------------
# Environment
# -----------------------------
def make_env(seed=0):
    def _init():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init

env = DummyVecEnv([make_env(SEED + i) for i in range(NUM_ENVS)])

# Wrap video recording around one environment
env = VecVideoRecorder(
    env,
    VIDEO_FOLDER,
    record_video_trigger=lambda step: step % VIDEO_FREQ == 0,
    video_length=200,
    name_prefix=RUN_NAME
)

# -----------------------------
# Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PPO("MlpPolicy", env, verbose=1, device=device)
callback = TensorboardCallback()

# -----------------------------
# Train
# -----------------------------
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# -----------------------------
# Save model
# -----------------------------
model.save(os.path.join(LOG_DIR, "ppo_model"))

# -----------------------------
# Evaluate and render
# -----------------------------
obs = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)  # ✅ Now returns 4 values thanks to the wrapper
    if done[0]:
        obs = env.reset()

env.close()
writer.close()
print(f"✅ Training complete!\nTensorBoard logs: {LOG_DIR}\nVideos: {VIDEO_FOLDER}")
