# Template

This repository is a template for AI research projects. 

## Environment Setup
All environment configurations are provided in ``requirements.txt``. To install all packages, you can create a conda environment and install the packages as follows: 
```bash
conda create -n NAME python=3.8
conda activate NAME
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

## Training & Evaluation
The config ``job_type`` is used to specify the type of job, which can be ``'train'``, ``'debug'``, or ``'test'``.  There are ``train.sh`` and ``test.sh`` in the ``scripts`` folder. You can run the following command to train or test the model:
```bash 
bash scripts/train.sh
bash scripts/test.sh
```
Remember to modify the ``ckpt_path`` in the ``test.sh`` before evaluation.

## Resume from Checkpoints 
You can use the config ``resume`` to specify the path of checkpoint to resume training.

## Star and Fork
If you find this project helpful, please consider star or fork this repository.

Star, fork and follow me for more projects!

