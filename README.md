# Car Racing AI with Track Limits

A reinforcement learning agent trained with PPO to drive around Gymnasium's CarRacing-v3, with custom F1-style track limit rules.

## The rules
- Touching the grass makes the lap invalid and loses 10 points, then 1 point every frame on the grass
- 3 seconds on the grass in a row ends the round
- Going too fast into sharp corners loses points
- A valid lap gives the AI 300 points

## How off-track detection works
The AI only sees a 96x96 picture of the game. The code checks the colour of the pixels around the car: green means grass, grey means road.

## Reward hacking
During training, the AI kept finding loopholes in the rules instead of driving properly:
- **Parking:** it learned that stopping avoided penalties
- **Stopping:** it gave up moving when penalties were too harsh
- **Corner cutting:** it went off track to save time
- **Giving up:** after one mistake it stopped trying

Each loophole meant changing the rules and retraining. Giving up was not fully solved, so the earlier, better-balanced model was kept.

## Results
The agent has completed a valid lap and reaches around 99% of the track. It isn't consistent yet: in the replay it crosses the line cleanly, then runs onto the grass afterwards, so the next lap shows as invalid.

## Built with
- Python 3.13
- Gymnasium (CarRacing-v3)
- Stable-Baselines3 (PPO)
- pygame-ce

## Run it
```
pip install -r requirements.txt
python watch.py
```

## Next steps
- Continuous controls for smoother driving
- Getting the AI to stay on track and complete valid laps consistently
- Racing against itself to set the fastest valid lap

## Notes
AI was used to help with explanations, debugging and documentation.
