# Offline RL Kit for Nuclear Fusion

## Requirements

- You need to download/clone this open-source repo: [dynamics-toolbox](https://github.com/LucasCJYSDL/dynamics-toolbox). 

- Make a virtual environment with python=3.9, then enter the repo above and run:
    ```bash
    pip install -r requirements.txt
    pip install -e .    
    ```

## Policy Learning

- Please start from converting the raw fusion data to the format required by offline RL or Imitation Learning:
    ```bash
    python rl_preparation/process_raw_data.py
    ```

- You can run different offline RL algorithms simply by:
    ```bash
    python rl_scripts/run_XXX.py --task YYY --seed Z
    ```
    - XXX can be one of [rombrl, cql, edac, combo, mobile, bambrl, rambo], corresponding to our algorithm and the 6 baselines used in the paper.
    - YYY can be one of [betan, dens_component1, rotation_component1], corresponding to the tracking tasks for betan, density, and rotation, respectively.
    - The random seed Z can be one of [0, 1, 2].

## Evaluations

- Plots of episodic returns during the policy learning process, along with trained policy model checkpoints, will be generated and saved in the 'log' folder.

## About the Fusion Data

The execution of these experiments relies on operational data from DIII-D, which is protected. As a result, we are unable to release them until we obtain the necessary approvals.

