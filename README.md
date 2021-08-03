# BAC-DAC
An OpenAI Gym toolkit for continuous control with Bayesian AC reinforcement learning.


![After 500 BAC policy updates](/500_updates.gif)
https://youtu.be/nkaAULbHVV4

## Pre-requisites
### Packages
1. NumPy, SciPy
2. gym.py (no Mujoco yet)
3. Pandas
4. CUDA Toolkit 11.3 (for gpu-accelerated branch)
5. CuPy for CUDA 11.3 (for gpu-accelerated branch)

### Hardware
1. At least Intel Core i3 3rd Gen (~ 1 hour simulation time for 500 BAC updates)
2. Dedicated Nvidia GPU with Compute Capability > 3.0 (https://developer.nvidia.com/cuda-gpus)

## Results

![MSE vs MAE](/MSE_vs_MAE.png)

![Avg. Batch Rewards](/avg_reward.png)

![Avg. Episode Lengths / Batch](/avg_length.png)

