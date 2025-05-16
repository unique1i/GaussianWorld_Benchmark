from dataclasses import dataclass, field
from typing import Tuple, Type
import torchvision
import torch
import open_clip

from torch import nn
from transformers import AutoProcessor, AutoModel

CROP_SIZE = 512


@dataclass
class SigLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: SigLIPNetwork)
    # Use the SigLIP model ID from Hugging Face
    clip_model_pretrained: str = "google/siglip2-base-patch16-512"  #
    # SigLIP default embedding dimension for the vision encoder is 768.
    clip_n_dims: int = 768  # 1152
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class SigLIPNetwork(nn.Module):
    def __init__(self, config: SigLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            self.config.clip_model_pretrained,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.config.clip_model_pretrained, use_fast=True
        )
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives

    @property
    def name(self) -> str:
        return "siglip_{}".format(self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def encode_image(self, input):
        # Use the torchvision transforms (or optionally the processor) to preprocess.
        inputs = self.processor(images=[input], return_tensors="pt").to("cuda")
        return self.model.get_image_features(**inputs)


class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = "laion2b_s34b_b88k"
        self.clip_n_dims = 512
        self.embedding_dim = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)

        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.negatives]
            ).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: 32768x512
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input, mask=None):
        # check input type is unit8
        input = input.float() / 255.0  # the new prepprocess script uses 0-255 range now
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

    def encode_text(self, text_list, device):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_max_across(self, sem):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        for j in range(n_phrases):
            probs = self.get_relevancy(sem, j)
            pos_prob = probs[..., 0]
            n_phrases_sims[j] = pos_prob
        relev = torch.stack(n_phrases_sims)
        return relev
