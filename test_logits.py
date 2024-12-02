from prismatic.vla import get_vla_dataset_and_collator
from prismatic.models.backbones.vision import ImageTransform
from prismatic.models.backbones.llm.prompting.qwen_prompter import QwenPromptBuilder
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np

num_extra_tokens = 256


class DummyImageTransform:
    def __call__(self, x):
        return torch.from_numpy(np.array(x))


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

added = tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(num_extra_tokens)])
assert added == num_extra_tokens, f"Added {added} of {num_extra_tokens} extra tokens to tokenizer!"
print(f"Added {num_extra_tokens} extra tokens.")

vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
    "/nfs/kun2/datasets/tfds",
    "bridge_dataset",
    image_transform=DummyImageTransform(),
    tokenizer=tokenizer,
    prompt_builder_fn=QwenPromptBuilder,
    default_image_resolution=(3, 224, 224),
    shuffle_buffer_size=1000,
    action_tokenizer="extra_action_tokenizer",
)

dataloader = DataLoader(
    vla_dataset,
    batch_size=256,
    collate_fn=collator,
    num_workers=0,
)

all_accuracies = []
i = 0
for batch in dataloader:
    i += 1
    print(i)
    action_gt = batch["labels"].numpy()
    action_preds = batch["logits"].numpy().argmax(axis=-1)
    mask = (action_tokenizer.action_token_end_idx > action_gt) & (
        action_gt > action_tokenizer.action_token_begin_idx
    )

    # Compute Accuracy
    accuracy = np.sum((action_preds == action_gt) & mask) / np.sum(mask)
    all_accuracies.append(accuracy)
    if i > 10:
        break
print(f"accuracy: {np.mean(all_accuracies)}")

