import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
from transformers.modeling_outputs import BaseModelOutput
from torch import Tensor
from typing import *


def transform_func(
    tokenizer: PreTrainedTokenizerFast, max_length: int, examples: Dict[str, List]
) -> BatchEncoding:
    return tokenizer(
        examples["contents"],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        truncation=True,
    )


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)(
                {k: _move_to_cuda(v) for k, v in maybe_tensor.items()}
            )
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def create_batch_dict(
    tokenizer: PreTrainedTokenizerFast, input_texts: List[str], max_length: int = 512
) -> BatchEncoding:
    return tokenizer(
        input_texts,
        max_length=max_length,
        padding=True,
        pad_to_multiple_of=8,
        return_token_type_ids=False,
        truncation=True,
        return_tensors="pt",
    )


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


class E5MultilingualWrapper(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name="intfloat/multilingual-e5-large-instruct",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.encoder.cuda()
        self.encoder.eval()
        self.query_instruct = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self.query_instruct}\nQuery: {query}"

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.get_detailed_instruct(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = [
            "{} {}".format(doc.get("title", ""), doc["text"]).strip() for doc in corpus
        ]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 64
        for start_idx in tqdm(
            range(0, len(input_texts), batch_size), desc="encoding", mininterval=10
        ):
            batch_input_texts: List[str] = input_texts[
                start_idx : start_idx + batch_size
            ]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.amp.autocast("cuda"):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = self.average_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)
