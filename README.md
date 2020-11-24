# IAR : Basic-Policy-Gradient-Labs

## Installation

### Main installs:
```
pip3 install -r requirements.txt
```

### Install gym

Actually, the main install above does it, but if you want to do everything manually...

```
pip3 install gym
```

More information here:

https://gym.openai.com/docs/#installation

### Install Continuous Cartpole Environment

```
pip3 install -e my_gym
```

And that should be it!

## Example of command

```
python3 main_pg --env_name Pendulum-v0 --nb_repet 1 --nb_cycles 500 --max_episode_steps 200 --policy_type squashedGaussian
```

The list of possible arguments is found in arguments.py, together with the default values# IAR_RL
