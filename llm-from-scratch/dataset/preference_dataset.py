import torch


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}")

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def decode_tokens_from_batch(token_ids, tokenizer):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.flatten().tolist()
    return tokenizer.decode(token_ids)


class PreferenceDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch,
                   pad_token_id=50256,
                   ignore_index=-100,
                   allowed_max_length=None,
                   mask_prompt_tokens=True,
                   device="cpu"):
        batch_data = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "rejected_mask": [],
            "chosen_mask": []
        }
        # Find the longest sequence in the batch to set the same padding length
        max_length_common = 0
        if batch:
            for key in ["chosen", "rejected"]:
                current_max = max(len(item[key]) + 1 for item in batch)
                max_length_common = max(max_length_common, current_max)

        for item in batch:
            prompt = torch.tensor(item["prompt"])
            batch_data["prompt"].append(prompt)

            for key in ["chosen", "rejected"]:
                sequence = item[key]
                padded = sequence + [pad_token_id
                                     ] * (max_length_common - len(sequence))
                mask = torch.ones(len(padded)).bool()

                mask[len(sequence):] = False

                # +2 sets the two newline characters ("\n") before "### Response" to False
                if mask_prompt_tokens:
                    mask[:prompt.shape[0] + 2] = False

                batch_data[key].append(torch.tensor(padded))
                batch_data[f"{key}_mask"].append(mask)

        for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
            tensor_stack = torch.stack(batch_data[key])

            if allowed_max_length is not None:
                tensor_stack = tensor_stack[:, :allowed_max_length]

            batch_data[key] = tensor_stack.to(device)

        return batch_data
