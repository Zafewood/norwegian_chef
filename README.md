
# Norwegian Chef 🧑‍🍳

## Introduction 🥘

Norwegian Chef is a repo containing the implementation of RL agent learning in the two-agent benchmark [Overcooked-AI environment](https://github.com/HumanCompatibleAI/overcooked_ai). The goal of the agent is to learn to solve simple rooms by learning actions in order to deliver soup in the evnironment.

## Installation ☑️


```
pip install -r "requirements.txt"
```

Note that this repo uses the PyPI package overcooked-ai, so it does not require you to download the overcooked-ai repo itself


## Code Structure Overview 🗺

`norwegian_chef` contains:

- `runs/`: tensorboard-enabled summaries of the runs

- `trained/`: pre-trained models, also where new models will be saved

- `hyperparam_search.sh`: a bash script used to search a hyperparameter grid

- `norwegian_chef_baseline_multi_school`: to train both agents simultaneously

- `norwegian_chef_baseline_school`: main code to train one of the agents (while keeping the other random)

- `norwegian_chef_school`: legacy file to train one of the agents (but without the basline implementation)

- `requirements.txt`: requirements to set up and run the environments and training

- `test_multi.py`: test file for testing two agent models together. You should manually modify the *test_model* parameter to fit your model policy name in the *trained* folder. the *_a0* and *_a1* is appended later.

- `test.py`: test file for testing a single agent model. You should manually modify the *test_model* parameter to fit your model policy name in the *trained* folder. The hyperparameters is also able to be modified inline.



## Example Usage

`norwegian_chef_baseline_school.py`:


```
python norwegian_chef_baseline_school.py --visual --track --save
```

runs the training of a reinforce-with-baseline model, visualizes it using pygame, tracks it to tensorboard

You can open tensorboard with `tensorboard --logdir=runs`
The pygame window opens to black by default but rendering is activated by pressing `p` and decativated by pressing `q`. This is so that the training can run efficient, since it slows down while rendering at the same time.

The model policy is saved in the *trained* folder named by the train_model parameter, as well as the time.

## Notes

Coding this project Github Copilot have been active, able to aid with boilerplate code snippets. The core implementation, methods and fine-tuning is still based on my own design, writing and debugging.