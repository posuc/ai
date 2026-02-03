from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.simple_game import SimpleGameEnv
import pymongo

app = FastAPI(title="Game AI PPO Service")

# --- MongoDB ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["game_ai"]
sessions = db["sessions"]

# --- Загрузка модели ---
def load_model(model_path="ppo_simple_game.zip"):
    env = DummyVecEnv([lambda: SimpleGameEnv()])
    model = PPO.load(model_path, env=env)
    return model, env

model, env = load_model()

class GameState(BaseModel):
    state: list[float]

class ActionResponse(BaseModel):
    action: int
    probas: list[float]

@app.post("/infer", response_model=ActionResponse)
def infer_action(game_state: GameState):
    state = np.array(game_state.state, dtype=np.float32).reshape(1, -1)
    try:
        action, _ = model.predict(state, deterministic=False)
        if isinstance(action, np.ndarray):
            action = action.item()

        probas = [0.1, 0.8, 0.1] if action == 1 else [0.3, 0.4, 0.3]

        # Логируем сессию в MongoDB
        sessions.insert_one({
            "state": game_state.state,
            "action": int(action),
            "probas": probas,
        })

        return ActionResponse(action=int(action), probas=probas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
