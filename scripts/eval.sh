dataset=cifar10
export CUDA_VISIBLE_DEVICES=0,1
source ~/anaconda3/bin/activate light
python engine/main.py \
    dataset=${dataset} \
    training=${dataset} \
    training.job_type='test' \
    training.exp_name="test" \
    training.num_workers=8 \
    training.batch_size=128 \
    training.val_check_interval=1 \
    training.lr=4e-4 \
    training.precision=32 \
    training.logger=wandb \
    ckpt_path='last.ckpt' \