"""

This file is used to train the car racing model using the PPO algorithm from Stable Baselines3.

"""

# Importing libraries
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from track_limits import make_env # Retrieves the function that builds the game from track_limits.py

env = make_env() # Builds the game with the track limits rules, black and white frames and the last 4 frames

# Saves the trained model every 20,000 steps and saves it to checkpoints/
checkpoint = CheckpointCallback(save_freq=20_000, save_path="checkpoints/")

# Loads the saved model to carry on training it
model = PPO.load(
    "car_ppo",
    env=env,
    tensorboard_log="logs/",
    custom_objects={
        "learning_rate": 0.0001, # Learning rate is set to 0.0001 to stop it from forgetting what it has already learned
        "lr_schedule": lambda _: 0.0001,
        "clip_range": lambda _: 0.1,
    },
)

# Trains the model a further 1,000,000 steps, using the checkpoint callback to save the model every 20,000 steps, and saves it to car_ppo
model.learn(total_timesteps=1_000_000, callback=checkpoint, reset_num_timesteps=False)
model.save("car_ppo")