import os

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')

import tiktoken
from dataset.spam_dataset import SpamDataset
from model.gpt2_model import (GPTModel, generate_text_greedy,
                              load_weights_into_gpt, text_to_token_ids,
                              token_ids_to_text)
from model.lora_layers import (LinearWithLoRA, LoRALayer,
                               replace_linear_with_lora)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tools.gpt_download import download_and_load_gpt2


class LightningGPT2SpamClassifierLoRA(pl.LightningModule):

    def __init__(self, gpt2_model, tokenizer, train_cfg):
        super().__init__()
        self.gpt2_model = gpt2_model
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg

    def forward(self, token_ids):
        return self.gpt2_model(token_ids)

    def training_step(self, batch, batch_idx):
        # Unpack the batch - batch is a tuple of (inputs, targets)
        inputs, targets = batch

        # Forward pass
        logits = self.forward(inputs)

        # Calculate loss (cross entropy)
        if self.train_cfg["use_average_embeddings"]:
            # Use average token embeddings for classification
            avg_embeddings = logits.mean(dim=1)
            loss = torch.nn.functional.cross_entropy(avg_embeddings, targets)
        else:
            # Use the last token logits for classification
            last_token_logits = logits[:, -1, :]  # Logits of last output token
            loss = torch.nn.functional.cross_entropy(last_token_logits,
                                                     targets)

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
        if self.train_cfg["use_average_embeddings"]:
            # Use average token embeddings for classification
            avg_embeddings = logits.mean(dim=1)
            loss = torch.nn.functional.cross_entropy(avg_embeddings, targets)
        else:
            # Use the last token logits for classification
            last_token_logits = logits[:, -1, :]  # Logits of last output token
            loss = torch.nn.functional.cross_entropy(last_token_logits,
                                                     targets)

        # Calculate predictions
        if self.train_cfg["use_average_embeddings"]:
            preds = torch.argmax(avg_embeddings, dim=1)
        else:
            preds = torch.argmax(last_token_logits, dim=1)

        # Calculate metrics
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (
            precision + recall) > 0 else 0

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)

        # Log text examples to TensorBoard (every 50 batches)
        if batch_idx % 50 == 0 and self.logger:
            # Sample a few examples from the batch
            num_samples = min(3, inputs.size(0))
            for i in range(num_samples):
                # Convert token IDs back to text
                text = token_ids_to_text(inputs[i].unsqueeze(0).cpu().numpy(),
                                         self.tokenizer)
                gt_label = "spam" if targets[i].item() == 1 else "ham"
                pred_label = "spam" if preds[i].item() == 1 else "ham"

                # Log to TensorBoard
                self.logger.experiment.add_text(
                    f"val_example_{batch_idx}_{i}",
                    f"**Text:** {text}\n\n**Ground Truth:** {gt_label}\n\n**Prediction:** {pred_label}",
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
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(csv_file="data/sms_spam_collection/train.csv",
                                max_length=None,
                                tokenizer=tokenizer)

    val_dataset = SpamDataset(
        csv_file="data/sms_spam_collection/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer)

    test_dataset = SpamDataset(csv_file="data/sms_spam_collection/test.csv",
                               max_length=train_dataset.max_length,
                               tokenizer=tokenizer)

    num_workers = 4
    batch_size = 8

    torch.manual_seed(123)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             drop_last=False,
                                             persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last=False,
                                              persistent_workers=True)

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

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`")

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="tools/gpt2_model_ckpts")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    # DEBUG
    # model.eval()
    # text_1 = "Every effort moves you"
    # token_ids = generate_text_greedy(
    #     model=model,
    #     idx=text_to_token_ids(text_1, tokenizer),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG["context_length"])
    # print(token_ids_to_text(token_ids, tokenizer))

    # DEBUG
    # # check if the model can classify spam
    # model.eval()
    # text_2 = (
    #     "Is the following text 'spam'? Answer with 'yes' or 'no':"
    #     " 'You are a winner you have been specially"
    #     " selected to receive $1000 cash or a $2000 award.'"
    # )
    # token_ids = generate_text_greedy(
    #     model=model,
    #     idx=text_to_token_ids(text_2, tokenizer),
    #     max_new_tokens=23,
    #     context_size=BASE_CONFIG["context_length"])
    # print(token_ids_to_text(token_ids, tokenizer))

    # Add classification head, which maps the embedding to a 2-class output rather than a vocab_size-class output
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],
                                     out_features=num_classes)

    total_params_before = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
    print(f"Total trainable parameters before: {total_params_before:,}")

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    total_params_after_freeze = sum(p.numel() for p in model.parameters()
                                    if p.requires_grad)
    print(
        f"Total trainable parameters after freeze: {total_params_after_freeze:,}"
    )

    # Replace linear layers with LoRA layers
    replace_linear_with_lora(model, rank=16, alpha=16)
    total_params_lora = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
    print(
        f"Total trainable LoRA parameters: {total_params_lora:,}, {total_params_lora * 1.0 / total_params_before:.4f}x"
    )

    TRAIN_CFG = {
        "learning_rate": 5e-5,
        "weight_decay": 0.1,
        "num_epochs": 10,
        "one_epoch_steps": len(train_loader),
        "use_average_embeddings":
        True,  # Use average token embeddings for classification or the last token
    }

    model = LightningGPT2SpamClassifierLoRA(model, tokenizer, TRAIN_CFG)

    exp_name = f"gpt2-sft-spam-classifier-avg_embed_{TRAIN_CFG['use_average_embeddings']}_lora_rank_16_alpha_16"
    tb_logger = TensorBoardLogger("lightning_logs/", name=exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        filename="{exp_name}-{epoch:02d}-{val_accuracy:.4f}",
        save_top_k=3,
        monitor="val_accuracy",
        mode="max")  # Use "max" mode since we want to maximize accuracy

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
    train_gpt()
