import os

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')

import tiktoken
from dataset.gpt_dataset import create_gpt_dataloader_v2
from model.gpt2_model import (GPTModel, generate_sample_text_greedy,
                              generate_sample_text_with_temperature_and_topk)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class LightningGPT2(pl.LightningModule):

    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.gpt_model_cfg = cfg["gpt_model_cfg"]
        self.train_cfg = cfg["train_cfg"]
        self.model = GPTModel(self.gpt_model_cfg)
        self.tokenizer = tokenizer

    def forward(self, token_ids):
        return self.model(token_ids)

    def training_step(self, batch, batch_idx):
        # Unpack the batch - batch is a tuple of (inputs, targets)
        inputs, targets = batch

        # Forward pass
        logits = self.forward(inputs)

        # Calculate loss (cross entropy)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.gpt_model_cfg["vocab_size"]),
            targets.reshape(-1))

        # Log training loss
        self.log("train_loss",
                 loss,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch - batch is a tuple of (inputs, targets)
        inputs, targets = batch

        # Forward pass
        logits = self.forward(inputs)

        # Calculate loss (cross entropy)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.gpt_model_cfg["vocab_size"]),
            targets.reshape(-1))

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Generate a sample text every few batches to see progress
        if batch_idx == 0:
            # generated_text = generate_sample_text_greedy(self.model, self.tokenizer, self.device, "Every effort moves you")
            generated_text = generate_sample_text_with_temperature_and_topk(
                self.model,
                self.tokenizer,
                self.device,
                "Every effort moves you",
                max_new_tokens=10,
                temperature=1.4,
                top_k=10,
            )
            self.logger.experiment.add_text("generated_text",
                                            f"{generated_text}",
                                            self.global_step)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"])

        # Store the total steps in the training config
        # We'll use an estimate based on dataset size and batch size
        # This will be passed from the train_gpt function
        total_steps = self.train_cfg.get("total_steps",
                                         self.train_cfg["num_epochs"] *
                                         100)  # Default fallback

        # OneCycleLR with warmup (first 10% of steps are for warmup)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.train_cfg["learning_rate"],
            total_steps=total_steps,
            pct_start=0.1,  # 10% of training for warmup
            div_factor=25,  # initial_lr = max_lr/div_factor
            final_div_factor=10000,  # final_lr = initial_lr/final_div_factor
            anneal_strategy='cos')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update at each step rather than epoch
            }
        }


def train_gpt(data_dir):
    pl.seed_everything(42)

    all_files = [
        os.path.join(path, name) for path, subdirs, files in os.walk(data_dir)
        for name in files if name.endswith((".txt"))
    ]
    total_files = len(all_files)
    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        return
    print("Total files:", total_files)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-key-value bias
    }

    TRAIN_CFG = {
        "learning_rate": 5e-3,
        "num_epochs": 1,
        "batch_size": 8,
        "weight_decay": 0.1
    }

    # Train/validation ratio
    train_ratio = 0.8
    split_idx = int(train_ratio * total_files)

    train_loader_cfg = {
        "batch_size": TRAIN_CFG["batch_size"],
        "max_length": GPT_CONFIG_124M["context_length"],
        "stride": GPT_CONFIG_124M["context_length"],
        "drop_last": True,
        "shuffle": True,
        "num_workers": 8
    }
    train_files = all_files[:split_idx]
    train_files_small = train_files[:3]
    train_loader = create_gpt_dataloader_v2(train_files_small,
                                            train_loader_cfg)

    # Calculate total steps for the scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * TRAIN_CFG["num_epochs"]

    # Add total_steps to the training config
    TRAIN_CFG["total_steps"] = total_steps

    val_loader_cfg = {
        "batch_size": TRAIN_CFG["batch_size"],
        "max_length": GPT_CONFIG_124M["context_length"],
        "stride": GPT_CONFIG_124M["context_length"],
        "drop_last": False,
        "shuffle": False,
        "num_workers": 8
    }
    val_files = all_files[split_idx:]
    val_files_small = val_files[:1]
    val_loader = create_gpt_dataloader_v2(val_files_small, val_loader_cfg)

    cfg = {"gpt_model_cfg": GPT_CONFIG_124M, "train_cfg": TRAIN_CFG}
    tokenizer = tiktoken.get_encoding("gpt2")
    model = LightningGPT2(cfg, tokenizer)

    # wandb_logger = WandbLogger(project="gpt2-training")
    tb_logger = TensorBoardLogger("lightning_logs/", name="gpt2-124m")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/gpt2-124m",
        filename="gpt2-124m-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    accelerator = "cpu"
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "cuda"
    trainer = pl.Trainer(
        max_epochs=TRAIN_CFG["num_epochs"],
        accelerator=accelerator,
        devices=1,
        logger=[tb_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gutenberg_data_dir",
                        type=str,
                        default="data/gutenberg_processed")
    args = parser.parse_args()
    gutenberg_data_dir = args.gutenberg_data_dir
    train_gpt(gutenberg_data_dir)
