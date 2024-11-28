from prismatic.vla import get_vla_dataset_and_collator
from prismatic.models.backbones.vision import ImageTransform
from prismatic.models.backbones.llm.prompting.qwen_prompter import QwenPromptBuilder
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np

class DummyImageTransform:
    def __call__(self, x):
        return torch.from_numpy(np.array(x))

vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
    "/nfs/kun2/datasets/tfds",
    "bridge_dataset",
    image_transform=DummyImageTransform(),
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
    prompt_builder_fn=QwenPromptBuilder,
    default_image_resolution=(3, 224, 224),
    shuffle_buffer_size=1000,
    action_tokenizer="extra_action_tokenizer"
)

dataloader = DataLoader(
    vla_dataset,
    batch_size=4,
    collate_fn=collator,
    num_workers=0,
)

for batch in dataloader:
    breakpoint()
