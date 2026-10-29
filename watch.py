"""

This file contains code to watch a trained model play the game.

"""

# Imports libraries
import pygame
from stable_baselines3 import PPO
from track_limits import make_env # Retrieves the function that builds the game from track_limits.py

env = make_env(render_mode="rgb_array", max_steps=10000) # Builds the game with a picture to draw and 10,000 steps so it doesn't get cut off

CHECKPOINT = "car_ppo_final" # When the file is run, the model will load the car_ppo_final checkpoint

model = PPO.load(CHECKPOINT) # Loads the model from the car_ppo_final checkpoint

# Sets up the pygame window, clock and font for the on-screen text
pygame.init()
window = pygame.display.set_mode((600, 400))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

# Runs 5 rounds of the game
for round_no in range(1, 6):
    obs, info = env.reset()
    done = False

    # Runs the game until the round ends or 10,000 steps are reached
    while not done:
        # The model predicts the action to take based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Updates the pygame window with the current game state
        picture = env.render()
        window.blit(pygame.surfarray.make_surface(picture.swapaxes(0, 1)), (0, 0))

        # Displays the current lap time and lap validity
        colour = (255, 255, 255) if info["lap_valid"] else (255, 0, 0)
        timer = font.render(f"{info['seconds']:.2f}s", True, colour)
        window.blit(timer, (10, 10))

        # Shows the lap validity status
        if not info["lap_valid"]:
            text = font.render("LAP INVALID", True, (255, 0, 0))
            window.blit(text, (600 - text.get_width() - 10, 10))

        # Shows the return to track warning
        if info["grass_frame_counter"] > 0:
            left = 3 - info["grass_frame_counter"] / 50
            warning = font.render(f"RETURN TO TRACK: {left:.1f}s", True, (255, 200, 0))
            window.blit(warning, ((600 - warning.get_width()) // 2, 40))

        # Shows the track coverage progress
        covered = env.unwrapped.tile_visited_count / len(env.unwrapped.track) * 100
        progress = font.render(f"Track covered: {covered:.0f}%", True, (255, 255, 255))
        window.blit(progress, (10, 35))

        # Updates the pygame display
        pygame.display.flip()
        pygame.event.pump()
        clock.tick(50)

env.close() # Closes the environment