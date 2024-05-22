# GTG

Official Code for Guided Trajectory Generation with Diffusion Models for Offline Model-based Optimization

### Environment Setup
To install dependencies, please run commands as follows:
```
# Create conda environment
conda create -n gtg python=3.8 -y
conda activate gtg

# Mujoco Installation
pip install Cython==0.29.36 numpy==1.22.0 mujoco_py==2.1.2.14
# Mujoco Compile
python -c "import mujoco_py"

# Torch Installation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Design-Bench Installation
pip install design-bench==2.0.12
pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
pip install botorch==0.6.4 gpytorch==1.6.0

# Decision Diffuser Installation
pip install jaynes==0.8.11 ml_logger==0.8.69
pip install gym==0.13.1 params_proto==2.9.6 scikit-image==0.17.2 scikit-video==1.1.11 scikit-learn==0.23.1 typed-argument-parser einops wandb
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# Download Design-Bench Offline Datasets: https://drive.google.com/file/d/11nAyb7_tmlGd0ri5aOK5YP3ZMIYFOmvS/view?usp=drive_link
unzip design_bench_data.zip
rm -rf design_bench_data.zip
mv -v design_bench_data <CONDA_PATH>/envs/gtg/lib/python3.8/site-packages
```

### Code references
Our implementation is based on "Is Conditional Generative Modeling is all you need for Decision Making?" ([https://github.com/anuragajay/decision-diffuser](https://github.com/anuragajay/decision-diffuser))

### Main Experiments
You can run the following commands to train and evaluate our method on Design-Bench tasks.

- Constructing Trajectories: To construct trajectories, you should run the following command.
```
python construct_trajectories.py --task <task>
```

- Training Models: To train models, you should run the following command.
```
python train.py --task <task> --horizon <horizon> --seed <seed>
```

- Evaluate: To sample candidates and do evalaution, you should run the following command.
```
python evaluate.py --task <task> --horizon <horizon> --ctx_len <ctx_len> --alpha <alpha> --seed <seed>
```

### Additional Experiments
You can run the following commands to train and evaluate our method on practical settings of Design-Bench tasks.

- Sparse Setting
```
python construct_trajectories.py --task <task> --frac <frac>

python train.py --task <task> --horizon <horizon> --seed <seed> --frac <frac>

python evaluate.py --task <task> --horizon <horizon> --ctx_len <ctx_len> --alpha <alpha> --seed <seed> --frac <frac>
```

- Noisy Setting
```
python construct_trajectories.py --task <task> --sigma <sigma>

python train.py --task <task> --horizon <horizon> --seed <seed> --sigma <sigma>

python evaluate.py --task <task> --horizon <horizon> --ctx_len <ctx_len> --alpha <alpha> --seed <seed> --sigma <sigma>
```
