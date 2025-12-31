Differences between CoinRun and other datasets
1. Collection method
CoinRun: Collected locally using collect_coinrun_data.py following the Genie paper specs.
Others (picodoom, pole_position, pong, sonic, zelda): Pre-collected and downloaded from HuggingFace (AlmondGod/tinyworlds).
2. Dataset size
CoinRun: 10,000,000 frames (10,000 levels × 1,000 timesteps per level)
picodoom: 59,785 frames
sonic: 41,242 frames
zelda: 72,410 frames
pong: ~14MB (small)
pole_position: ~17MB (small)
CoinRun is ~100–200x larger than the others.
3. Resolution
CoinRun: 160×90×3 (matches the paper)
picodoom: 64×64×3
sonic: 64×64×3
zelda: 128×128×3
The others use different resolutions.
4. Collection parameters (known for CoinRun)
Environment: Procgen CoinRun
Difficulty: "hard"
Policy: Random with no action repeats
Levels: 0–10,000
Timesteps per level: 1,000
Total: 10M transitions
For the other datasets, these parameters are unknown.
5. Why CoinRun was chosen (from the paper)
Thousands of visually diverse levels
Simple platformer-like dynamics
"Hard" mode for diversity
Large scale (10M transitions)
Implications for training
Resolution mismatch: The others need resizing to 160×90 to match CoinRun.
Size difference: CoinRun provides most of the data.
Collection policy: The others may have been collected with different policies (e.g., trained agents, different action distributions).
Use together: You can combine them, but:
Resize all to 160×90
Be aware of potential distribution differences
CoinRun will dominate due to size