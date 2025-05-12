# Policy-Driven World Model Adaptation for Robust Offline Model-based Reinforcement Learning

- Please download the dynamics/reward models and hyperparameter files from [d4rl_data](https://drive.google.com/drive/folders/1FiJbpAJvul629u4VjgOyugHwcBPJyc7u?usp=sharing) to the folder 'data'.

- Please set up a virtue environment based on the instructions from [OfflineRLKit](https://github.com/yihaosun1124/OfflineRL-Kit).

- You can simly reproduce the performance of different algorithms (as in Table 1) by running:
    ```bash
    python run_XXX.py
    ```
    - XXX can be one of [rombrl2, cql, edac, combo, rambo, mobile, bambrl], which corresponds to our algorithm and the 6 baselines used in the paper.
    - To specify the task and random seed for each run, you can simply change the 'load_path_id' at the bottom of each 'run_XXX.py' script.