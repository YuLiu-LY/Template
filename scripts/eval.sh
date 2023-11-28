dataset=cifar10
model=my
export CUDA_VISIBLE_DEVICES=0,1
source ~/anaconda3/bin/activate AGI
python main.py \
    dataset=${dataset} \
    model=${model} \
    training=${model}_${dataset} \
    ckpt_path='/home/yuliu/Projects/Template/runs/cifar10/test_11281319/checkpoints/last.ckpt' \
    training.job_type=test \
    training.exp_name=eval_${model} \
    training.batch_size=128 \
    training.val_check_interval=1 \
    training.lr=4e-4 \
    training.precision=32 \
    # training.logger=wandb \