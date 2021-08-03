# BAC-DAC
An OpenAI Gym toolkit for continuous control with Bayesian AC reinforcement learning.


![After 500 BAC policy updates](/500_updates.gif) <br/>
https://youtu.be/nkaAULbHVV4 <br/>

Run ```sample/mountain_car_v0_no_jupyter.py``` <br/>

^^Notice - Working on CUDA accelerated branch. I will update it here ASAP.

## Pre-requisites
### Packages
1. NumPy, SciPy
2. OpenAI Gym gym.py (no Mujoco yet)
3. Pandas, matplotlib
4. CUDA Toolkit 11.3 (for gpu-accelerated branch)
5. CuPy for CUDA 11.3 (for gpu-accelerated branch)

### Hardware
1. At least Intel Core i3 3rd Gen (~ 1 hour simulation time for 500 BAC updates)
2. At least 4 GB DDR3 RAM
3. (only for GPU branch) Dedicated Nvidia GPU with Compute Capability > 3.0 (https://developer.nvidia.com/cuda-gpus)

## Results
#### 5 episodes per batch <<br/>
![MSE vs MAE](/MSE_vs_MAE.png)

![Avg. Batch Rewards](/avg_reward.png)

![Avg. Episode Lengths / Batch](/avg_length.png)

### Thoughts
We see that it smoothly achieves the goal. Since this is control control, ```action_space = [-1.0, 1.0]```. The agents above is more inclined to take ```action ~= 1.0```. Running the sim for higher BAC updates would probably see the agent figure out how to take ```action ~= -1.0``` once it is up-slope towards the GOAL. Currently, the sim is processor heavy, thus slow. I am working on CUDA acceleration to speed up the NumPy and SciPy operations.

## References
1. Ghavamzadeh, Mohammad, Yaakov Engel, and Michal Valko. "Bayesian policy gradient and actor-critic algorithms." The Journal of Machine Learning Research 17.1 (2016): 2319-2371. **Main ref**
2. Ghavamzadeh, Mohammad, and Yaakov Engel. "Bayesian actor-critic algorithms." Proceedings of the 24th international conference on Machine learning. 2007.
3. Ciosek, Kamil, et al. "Better exploration with optimistic actor-critic." arXiv preprint arXiv:1910.12807 (2019).
4. Ghavamzadeh, Mohammad, et al. "Bayesian reinforcement learning: A survey." arXiv preprint arXiv:1609.04436 (2016).
5. Kurenkov, Andrey, et al. "Ac-teach: A bayesian actor-critic method for policy learning with an ensemble of suboptimal teachers." arXiv preprint arXiv:1909.04121 (2019).
6. Bhatnagar, Shalabh, et al. "Natural actor–critic algorithms." Automatica 45.11 (2009): 2471-2482.
