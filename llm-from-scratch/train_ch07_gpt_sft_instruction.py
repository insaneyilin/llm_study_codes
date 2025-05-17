import json
import os
from functools import partial

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')

import tiktoken
from dataset.instruction_dataset import InstructionDataset, format_input
from model.gpt2_model import (GPTModel,
                              generate_sample_text_with_temperature_and_topk,
                              load_weights_into_gpt, text_to_token_ids,
                              token_ids_to_text)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tools.gpt_download import download_and_load_gpt2
from tqdm import tqdm


class LightningGPT2InstructionSFT(pl.LightningModule):

    def __init__(self, gpt2_model, tokenizer, train_cfg):
        super().__init__()
        self.model = gpt2_model
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg

    def forward(self, token_ids):
        return self.model(token_ids)

    def training_step(self, batch, batch_idx):
        # Unpack the batch - batch is a tuple of (inputs, targets)
        inputs, targets = batch

        # Forward pass
        logits = self.forward(inputs)

        # Calculate loss (cross entropy)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.tokenizer.n_vocab), targets.reshape(-1))

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
            logits.reshape(-1, self.tokenizer.n_vocab), targets.reshape(-1))

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Generate a sample text every few batches to see progress
        if batch_idx == 0:
            pad_token_id = self.tokenizer.n_vocab - 1
            instruction = inputs[0]
            mask = instruction == pad_token_id  # shape: (seq_len, )
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                instruction = instruction[:indices[
                    0]]  # remove the padding token
            instruction = token_ids_to_text(instruction, self.tokenizer)
            generated_text = generate_sample_text_with_temperature_and_topk(
                self.model,
                self.tokenizer,
                self.device,
                start_context=instruction,
                max_new_tokens=10,
                temperature=1.4,
                top_k=10,
                eos_id=pad_token_id,
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
        total_steps = self.train_cfg["num_epochs"] * self.train_cfg[
            "one_epoch_steps"]

        # OneCycleLR with warmup (first 10% of steps are for warmup)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.train_cfg["learning_rate"],
        #     total_steps=total_steps,
        #     pct_start=0.1,  # 10% of training for warmup
        #     anneal_strategy='cos')

        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update at each step rather than epoch
            }
        }


def train_gpt():
    pl.seed_everything(42)
    ########################################
    # Load data
    ########################################
    file_path = "data/instruction-data.json"
    data = None
    with open(file_path, 'r') as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50 * "-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print("Device:", device)
    print(50 * "-")

    customized_collate_fn = partial(InstructionDataset.collate_fn,
                                    device=device,
                                    allowed_max_length=1024)

    # FIXME: The collate_fn seems to be not working properly when num_workers > 0.
    # Maybe due to PyTorch Lightning? https://github.com/pytorch/pytorch/issues/87688
    num_workers = 0
    batch_size = 8

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    ########################################
    # Load model
    ########################################

    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {
            "emb_dim": 768,
            "n_layers": 12,
            "n_heads": 12
        },
        "gpt2-medium (355M)": {
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16
        },
        "gpt2-large (774M)": {
            "emb_dim": 1280,
            "n_layers": 36,
            "n_heads": 20
        },
        "gpt2-xl (1558M)": {
            "emb_dim": 1600,
            "n_layers": 48,
            "n_heads": 25
        },
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="tools/gpt2_model_ckpts")

    gpt2_model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(gpt2_model, params)

    print("Loaded model:", CHOOSE_MODEL)
    print(50 * "-")

    # NOTE: Here we do not freeze the model, and we do not add a classification head.

    TRAIN_CFG = {
        "learning_rate": 5e-5,
        "weight_decay": 0.1,
        "num_epochs": 2,
        "one_epoch_steps": len(train_loader),
    }

    model = LightningGPT2InstructionSFT(gpt2_model, tokenizer, TRAIN_CFG)

    exp_name = "gpt2-instruction_sft"
    tb_logger = TensorBoardLogger("lightning_logs/", name=exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        filename="exp_name-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss")

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

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)
        generated_text = generate_sample_text_with_temperature_and_topk(
            model.model.to("cpu"),
            tokenizer,
            device="cpu",
            start_context=input_text,
            max_new_tokens=256,
            temperature=1.4,
            top_k=10,
            eos_id=tokenizer.n_vocab - 1,
        )

        response_text = generated_text[len(input_text):].replace(
            "### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-sft-response.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")


if __name__ == "__main__":
    train_gpt()
