
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import hydra
import torch
from engine import create_trainer
from engine.method import Method


@hydra.main(config_path='./configs', config_name='config', version_base='1.2')
def main(cfg):
    trainer = create_trainer(cfg)
    lightning_model = Method(cfg)
    
    if cfg.training.job_type == 'test':
        ckpt = torch.load(cfg.ckpt_path) 
        lightning_model.load_state_dict(ckpt['state_dict'])
        print(f'Load from checkpoint: {cfg.ckpt_path}')
        trainer.test(lightning_model)
    else:
        if trainer.logger is not None and cfg.watch_model:
            trainer.logger.watch(lightning_model)
        trainer.fit(lightning_model, ckpt_path=cfg.training.resume)

if __name__ == "__main__":
    main()