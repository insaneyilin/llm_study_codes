import json
import os
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')

import tiktoken
from dataset.preference_dataset import (PreferenceDataset,
                                        decode_tokens_from_batch, format_input)
from model.gpt2_model import (GPTModel, assign,
                              generate_sample_text_with_temperature_and_topk,
                              load_weights_into_gpt, text_to_token_ids,
                              token_ids_to_text)
from model.lora_layers import (LinearWithLoRA, LoRALayer,
                               replace_linear_with_lora)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tqdm import tqdm

GPT2_FINETUNED_MODEL = "checkpoints/gpt2-medium355M-sft-lora.pth"


def load_gpt2_model(gpt2_model, params):
    state_dict = {}
    for key, value in params.items():
        if key.startswith('model.'):
            # Remove the 'model.' prefix
            new_key = key[6:]  # Skip the first 6 characters ('model.')
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    # Load the state dictionary into the model
    gpt2_model.load_state_dict(state_dict)
    return gpt2_model


class LightningGPT2InstructionDPO(pl.LightningModule):

    def __init__(self, policy_model, reference_model, tokenizer, train_cfg):
        super().__init__()
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.train_cfg = train_cfg
        # Save hyperparameters for better checkpointing
        self.save_hyperparameters(ignore=['policy_model', 'reference_model'])

        # Freeze the reference model to ensure it's not updated during training
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def forward(self, token_ids):
        return self.policy_model(token_ids)

    def training_step(self, batch, batch_idx):
        # For DPO training, batch contains chosen and rejected responses
        # Compute DPO loss using the policy and reference models
        loss, chosen_rewards, rejected_rewards = self._compute_dpo_loss_batch(
            batch=batch, beta=self.train_cfg.get("beta", 0.1))

        # Get batch size from the input batch
        batch_size = batch["chosen"].size(0)

        # Log training metrics with explicit batch_size
        self.log("train/loss",
                 loss,
                 prog_bar=True,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size)
        self.log("train/chosen_reward",
                 chosen_rewards,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size)
        self.log("train/rejected_reward",
                 rejected_rewards,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size)
        self.log("train/reward_gap",
                 chosen_rewards - rejected_rewards,
                 on_step=True,
                 on_epoch=True,
                 batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        # Compute DPO loss for validation data
        loss, chosen_rewards, rejected_rewards = self._compute_dpo_loss_batch(
            batch=batch, beta=self.train_cfg.get("beta", 0.1))

        # Get batch size from the input batch
        batch_size = batch["chosen"].size(0)

        # Log validation metrics with explicit batch_size
        self.log("val/loss",
                 loss,
                 prog_bar=True,
                 on_epoch=True,
                 batch_size=batch_size)
        self.log("val/chosen_reward",
                 chosen_rewards,
                 on_epoch=True,
                 batch_size=batch_size)
        self.log("val/rejected_reward",
                 rejected_rewards,
                 on_epoch=True,
                 batch_size=batch_size)
        self.log("val/reward_gap",
                 chosen_rewards - rejected_rewards,
                 on_epoch=True,
                 batch_size=batch_size)

        # Generate a sample text every few batches to see progress
        if batch_idx == 0 and self.global_step % 100 == 0 and "prompt" in batch:
            pad_token_id = self.tokenizer.n_vocab - 1
            prompt = batch["prompt"][0]  # Take the first prompt in the batch

            # Convert tokens to text
            prompt_text = token_ids_to_text(prompt, self.tokenizer)

            # Generate text using the policy model
            generated_text = generate_sample_text_with_temperature_and_topk(
                self.policy_model,
                self.tokenizer,
                self.device,
                start_context=prompt_text,
                max_new_tokens=100,
                temperature=1.0,
                top_k=10,
                eos_id=pad_token_id,
            )

            # Log the generated text
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.add_text("generated_text",
                                                f"{generated_text}",
                                                self.global_step)

        return loss

    def configure_optimizers(self):
        # Only optimize the policy model parameters
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(
            ),  # Only pass policy_model parameters
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"])

        # Calculate total steps for the scheduler
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

    def _compute_dpo_loss(
        self,
        model_chosen_logprobs,
        model_rejected_logprobs,
        reference_chosen_logprobs,
        reference_rejected_logprobs,
        beta=0.1,
    ):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            model_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            model_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5.
                  Controls the strength of the regularization. We ignore the reference model as beta -> 0.

        Returns:
            A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
        """
        # Calculate log ratios between policy and reference models
        model_logratios = model_chosen_logprobs - model_rejected_logprobs
        reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs

        # The core DPO logits calculation
        logits = model_logratios - reference_logratios

        # DPO loss calculation, see https://arxiv.org/pdf/2305.18290.pdf equation 7
        losses = -F.logsigmoid(beta * logits)

        # Calculate implicit rewards for monitoring (detached to prevent gradients)
        chosen_rewards = (model_chosen_logprobs -
                          reference_chosen_logprobs).detach()
        rejected_rewards = (model_rejected_logprobs -
                            reference_rejected_logprobs).detach()

        # Return batch mean values
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def _compute_logprobs(self, logits, labels, selection_mask=None):
        """
        Compute per-token log probabilities and average them for each sequence.

        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size) - model output logits
            labels: Tensor of shape (batch_size, seq_len) - token IDs of the target sequence
            selection_mask: Optional tensor of shape (batch_size, seq_len) - mask for tokens to include in loss

        Returns:
            mean_log_prob: Mean log probability for each sequence in the batch, shape (batch_size,)
        """
        # Shift labels to align with next-token prediction (labels are the targets)
        labels = labels[:, 1:].clone()

        # Truncate logits to match the shifted labels length
        logits = logits[:, :-1, :]

        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather the log probabilities corresponding to the target tokens
        selected_log_probs = torch.gather(
            input=log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        if selection_mask is not None:
            # Shift mask to align with the shifted labels
            mask = selection_mask[:, 1:].clone()

            # Apply mask to only consider relevant tokens
            selected_log_probs = selected_log_probs * mask

            # Calculate mean log probability per sequence, avoiding division by zero
            mask_sum = mask.sum(-1)
            # Add small epsilon to prevent division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-10)
            avg_log_prob = selected_log_probs.sum(-1) / mask_sum

            return avg_log_prob
        else:
            # If no mask provided, average over all tokens
            return selected_log_probs.mean(-1)

    def _compute_dpo_loss_batch(self, batch, beta=0.1):
        """Compute the DPO loss on an input batch

        Args:
            batch: Dictionary containing 'chosen' and 'rejected' token sequences and optional masks
            beta: Temperature parameter for the DPO loss

        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards)
        """
        # Get log probabilities for chosen responses
        policy_chosen_log_probas = self._compute_logprobs(
            logits=self.policy_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch.get(
                "chosen_mask")  # Use get() to handle missing keys
        )

        # Get log probabilities for rejected responses
        policy_rejected_log_probas = self._compute_logprobs(
            logits=self.policy_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch.get("rejected_mask"))

        # Compute reference model log probabilities (without gradients)
        with torch.no_grad():
            ref_chosen_log_probas = self._compute_logprobs(
                logits=self.reference_model(batch["chosen"]),
                labels=batch["chosen"],
                selection_mask=batch.get("chosen_mask"))

            ref_rejected_log_probas = self._compute_logprobs(
                logits=self.reference_model(batch["rejected"]),
                labels=batch["rejected"],
                selection_mask=batch.get("rejected_mask"))

        # Compute the final DPO loss
        loss, chosen_rewards, rejected_rewards = self._compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas,
            beta=beta)

        return loss, chosen_rewards, rejected_rewards


def train_gpt():
    pl.seed_everything(42)
    ########################################
    # Load data
    ########################################
    file_path = "data/instruction-data-with-preference.json"
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
    # It seems MPS has memory leak issue, so we use CPU for now.
    # https://github.com/pytorch/pytorch/issues/145374
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    print("Device:", device)
    print(50 * "-")

    customized_collate_fn = partial(
        PreferenceDataset.collate_fn,
        device=device,
        allowed_max_length=1024,
        mask_prompt_tokens=True,
    )

    num_workers = 0
    batch_size = 8

    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             collate_fn=customized_collate_fn,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    finetuned_model_path = Path(GPT2_FINETUNED_MODEL)
    checkpoint = torch.load(finetuned_model_path)
    # Check loaded checkpoint
    # print(checkpoint['state_dict'].keys())

    CHOOSE_MODEL = "gpt2-medium (355M)"

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

    gpt2_model = GPTModel(BASE_CONFIG)
    # Freeze the model
    for param in gpt2_model.parameters():
        param.requires_grad = False

    # Replace linear layers with LoRA layers
    replace_linear_with_lora(gpt2_model, rank=16, alpha=16)

    load_gpt2_model(gpt2_model, checkpoint['state_dict'])

    # Test loaded finetuned model
    # gpt2_model.eval()
    # test_input_json = {
    #     "instruction": "What is the capital of China?",
    #     "input": "",
    # }
    # test_input_text = format_input(test_input_json)
    # with torch.no_grad():
    #     gpt2_model_cpu = gpt2_model.to("cpu")
    #     pad_token_id = tokenizer.n_vocab - 1
    #     test_output_text = generate_sample_text_with_temperature_and_topk(
    #         gpt2_model_cpu,
    #         tokenizer,
    #         "cpu",
    #         test_input_text,
    #         max_new_tokens=256,
    #         temperature=1.4,
    #         top_k=10,
    #         eos_id=pad_token_id)
    #     print(test_output_text)

    gpt2_model.to(device)

    # DPO needs two models: policy_model and reference_model
    policy_model = gpt2_model
    # Reference model is the same as the policy model, but with frozen parameters
    reference_model = GPTModel(BASE_CONFIG)
    # Replace linear layers with LoRA layers
    replace_linear_with_lora(reference_model, rank=16, alpha=16)
    reference_checkpoint = torch.load(GPT2_FINETUNED_MODEL,
                                      map_location=torch.device("cpu"))
    load_gpt2_model(reference_model, reference_checkpoint['state_dict'])
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    policy_model.to(device)
    reference_model.to(device)

    total_params_policy_model = sum(p.numel()
                                    for p in policy_model.parameters()
                                    if p.requires_grad)
    print(
        f"Total trainable parameters of policy model: {total_params_policy_model:,}"
    )

    total_params_reference_model = sum(p.numel()
                                       for p in reference_model.parameters()
                                       if p.requires_grad)
    print(
        f"Total trainable parameters of reference model: {total_params_reference_model:,}"
    )

    TRAIN_CFG = {
        "learning_rate": 5e-6,
        "weight_decay": 0.1,
        "num_epochs": 1,
        "one_epoch_steps": len(train_loader),
        "beta":
        0.1,  # DPO loss parameter, larger beta means the dpo loss can represent the preference more closely
    }

    model = LightningGPT2InstructionDPO(policy_model, reference_model,
                                        tokenizer, TRAIN_CFG)

    exp_name = "gpt2-instruction_sft_dpo-lora"
    tb_logger = TensorBoardLogger("lightning_logs/", name=exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{exp_name}",
        filename="epoch={epoch:02d}-val_loss={val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    accelerator = "cpu"
    # if torch.backends.mps.is_available():
    #     accelerator = "mps"
    # elif torch.cuda.is_available():
    #     accelerator = "cuda"
    print(f"Using accelerator: {accelerator}")
    trainer = pl.Trainer(max_epochs=TRAIN_CFG["num_epochs"],
                         accelerator=accelerator,
                         devices=1,
                         logger=[tb_logger],
                         callbacks=[checkpoint_callback, lr_monitor],
                         log_every_n_steps=1,
                         gradient_clip_val=1.0)

    trainer.fit(model, train_loader, val_loader)

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)
        generated_text = generate_sample_text_with_temperature_and_topk(
            model.policy_model.to("cpu"),
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
        test_data[i]["policy_model_response"] = response_text

        generated_text = generate_sample_text_with_temperature_and_topk(
            model.reference_model.to("cpu"),
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
        test_data[i]["reference_model_response"] = response_text

    test_data_path = "instruction-data-with-response-dpo.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")


if __name__ == "__main__":
    train_gpt()
