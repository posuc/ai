from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.simple_game import SimpleGameEnv
import os

def make_env():
    return SimpleGameEnv()

def train_model(total_timesteps=10_000, model_path="ppo_simple_game.zip"):
    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"✅ Модель сохранена: {model_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_model()
