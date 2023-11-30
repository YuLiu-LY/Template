dataset=cifar10
model=my
export CUDA_VISIBLE_DEVICES=2,3
source ~/anaconda3/bin/activate AGI
python main.py \
    dataset=${dataset} \
    model=${model} \
    training=${model}_${dataset} \
    training.job_type=train \
    training.exp_name=exp1 \
    training.batch_size=64 \
    training.val_check_interval=5 \
    training.lr=4e-4 \
    training.precision=32 \
    # training.logger=wandb \
