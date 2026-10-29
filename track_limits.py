"""

This project was made to learn about reinforcement learning through the use of Gymnasium and Python.
The aim of this project was to create a reinforcement learning agent that was able to travel around different racing circuits without going off the road as fast as possible.
The agent was trained with the PPO algorithm and is currently still under development.

"""

# Imported libraries used in the project.
# Gymnasium is a library for reinforcement learning environments
# Math is a library for mathematical functions.
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, FrameStackObservation
import math


class TrackLimits(gym.Wrapper): # A class that wraps the environment to add track limits

    def reset(self, **kwargs): # Resets the environment which helps start a new round
        obs, info = self.env.reset(**kwargs)
        self.frame_counter = 0 # Used for time and skips the camera zoom at the start of the round
        self.grass_frame_counter = 0 # Counts how many frames the car has been on the grass in a row
        self.lap_valid = True  # Stays True if the car never touches the grass this lap, becomes False if it does
        self.track_covered = 0 # Counts how much of the track the car has covered
        self.frames_without_progress = 0 # If the car hasn't made progress in 2 seconds, it ends the round
        return obs, info

    def sharp_corner_ahead(self): # This method is used to check if there is a sharp corner ahead
        track = self.env.unwrapped.track # Lists all the track pieces and their directions
        car_x, car_y= self.env.unwrapped.car.hull.position # Tracks the car's position
        current_road = min(range(len(track)), key=lambda i: (track[i][2] - car_x) ** 2 + (track[i][3] - car_y) ** 2) # Tracks the current road piece the car is on
        direction_now = track[current_road][1] # Tracks the direction of the current road piece
        direction_ahead = track[(current_road + 10) % len(track)][1] # Tracks the direction of the road piece 10 pieces ahead
        turn = abs((direction_ahead - direction_now + math.pi) % (2 * math.pi) - math.pi) # Works out how much the road ahead bends between the current road piece and the road piece 10 pieces ahead
        return turn > 0.8 # Returns True if the turn is sharper than 0.8 radians

    def step(self, action): # Takes the action the agent has chosen and returns the new observation, reward, terminated, truncated, and info
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_counter += 1

        if self.frame_counter >= 60: # Used 60 as it skips the camera zoom at the start of the track

            # This section of code checks if the area around the car is grass or road
            grass = 0
            road = 0

            for row in range(60, 84): # Checks the pixels around the car to see if they are grass or road, Stops before it reaches the info bar at row 84
                for col in range(43, 54):
                    r, g, b = obs[row, col]
                    r, g, b = int(r), int(g), int(b) # Turns the colours into normal numbers so the maths below works properly
                    if r > g + 50: # If the red value is more than 50 higher than the green value, it is the car or a red kerb
                        continue
                    elif g > r + 50: # Else if the green value is more than 50 higher than the red value, it is grass
                        grass += 1
                    else: # Else it is road
                        road += 1

            on_grass = grass > road # If grass is greater than road, the car is on grass meaning it is off the track

            # The first time the car goes onto the grass, the lap becomes invalid, and the AI will lose 10 points
            if on_grass and self.lap_valid:
                self.lap_valid = False
                reward -= 10

            # Everytime the car goes on the grass, it will lose 1 point every frame and the grass frame counter will increase by 1
            if on_grass:
                reward -= 1
                self.grass_frame_counter += 1

            # If the car is not on grass, it will reset the grass frame counter
            else:
                self.grass_frame_counter = 0

            # If the car is on grass for 150 frames (3 seconds) in a row, it will terminate the episode
            if self.grass_frame_counter >= 150:
                terminated = True

        reward -= self.corner_speed_penalty() # Car will lose points for going too fast into a sharp corner


        # This section of code checks to see if the car has covered more of the track than it has before.
        # If it hasn't for 2 seconds, then it will terminate the episode
        tiles_now = self.env.unwrapped.tile_visited_count
        if tiles_now > self.track_covered:
            self.track_covered = tiles_now
            self.frames_without_progress = 0
        else:
            self.frames_without_progress += 1

        if self.frames_without_progress >= 100:
            terminated = True

        # If the car has gone around the track without touching the grass, give the car 300 points
        if info.get("lap_finished") and self.lap_valid:
            reward += 300
            info["lap_time"] = self.frame_counter / 50
            terminated = True

        # Sends live details to watch.py every frame: the timer, if the lap is valid, and the grass countdown
        info["lap_valid"] = self.lap_valid
        info["seconds"] = self.frame_counter / 50
        info["grass_frame_counter"] = self.grass_frame_counter

        return obs, reward, terminated, truncated, info

    # Works out how many points to lose for going too fast into a sharp corner
    def corner_speed_penalty(self):
        car = self.env.unwrapped.car.hull
        speed = (car.linearVelocity[0] ** 2 + car.linearVelocity[1] ** 2) ** 0.5
        if speed > 45 and self.sharp_corner_ahead():
            return (speed - 45) * 0.02
        return 0

# Builds the game the same way for both training and watching
def make_env(render_mode=None, max_steps=5000):
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=False, max_episode_steps=max_steps)
    env = TrackLimits(env)
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)
    return env