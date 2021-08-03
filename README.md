# BAC-DAC
An OpenAI Gym toolkit for continuous control with Bayesian AC reinforcement learning.


![After 500 BAC policy updates](/500_updates.gif) <br/>
https://youtu.be/nkaAULbHVV4 <br/>

Run ```sample/mountain_car_v0_no_jupyter.py```

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

### Thoughts
We see that it smoothly achieves the goal. Since this is control control, ```action_space = [-1.0, 1.0]```. The agents above is more inclined to take ```action ~= 1.0```. Running the sim for higher BAC updates would probably see the agent figure out how to take ```action ~= -1.0``` once it is up-slope towards the GOAL. Currently, the sim is processor heavy, thus slow. I am working on CUDA acceleration to speed up the NumPy and SciPy operations.
