# -*- coding: utf-8 -*-

import os, random, time
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import PreTrainedTokenizerFast
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dual_datamodule import *
from dual_learner import *


@hydra.main(config_path = '../conf', config_name = 'config')
def main(cfg):
    seed_everything(cfg.seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
    tokenizer.add_special_tokens({'sep_token' : '<sep>'})

    datamodule = DataModule(cfg.datamodule, tokenizer)
    train_loader, val_loader, test_loader = datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()
    learner = Learner(cfg.learner, tokenizer)
    
    if cfg.wandb == False:
        os.environ["WANDB_MODE"] = "offline"
        
    if not os.path.exists(cfg.wandb_dir):
        os.makedirs(cfg.wandb_dir)
    logger = WandbLogger(
        name = f"{cfg.method}_{cfg.description}",
        save_dir = cfg.wandb_dir,
        project = f"NRF",
        entity = 'jeankim941',
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath = cfg.experiment_dir,
        monitor = 'val_loss',
        save_top_k = 1,
        mode = 'min'
    )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = cfg.patience,
        mode = 'min'
    )

    trainer = Trainer(
        accelerator = 'gpu',
        devices = [cfg.devices],
        logger = logger,
        max_steps = cfg.num_training_steps,
        accumulate_grad_batches = cfg.gradient_accumulation_steps,
        max_epochs = -1,
        callbacks = [checkpoint_callback, early_stopping]
    )

    trainer.fit(
        learner,
        train_dataloaders = train_loader,
        val_dataloaders = val_loader
    )

    trainer.test(
        model = learner,
        dataloaders = test_loader,
        ckpt_path = cfg.ckpt_path  
    )

if __name__ == '__main__':
    main()        