# BA_RFE
This repository was created as part of my bachelor thesis and provides the TD3 algorithm implemented in Tensorflow 2 
and can be used to design and test reward functions in PyBullet or OpenAI-gym environments.
Only the paths need to be adjusted based on the local file system of the used computer.

Requirements are an Python 3 environment with the packages: 
- Tensorflow 2 (pip install)
- gym          (pip install)
- pybullet
- pybullet-gym (clone from github repositoryhttps://github.com/benelot/pybullet-gym.git)
- numpy
- os

By setting the train variable in the main to True or False, models can be trained or tested.
For using OpenAI-gym environments line 104 (action = action[0]) in main.py needs to be commented out.
