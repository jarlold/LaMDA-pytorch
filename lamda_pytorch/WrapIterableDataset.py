from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, default_data_collator
from config.config import CFG

import copy
from itertools import chain
from typing import Union

from sentencepiece import SentencePieceProcessor
from torch.distributed import get_world_size
args = CFG
def tokenize_example(tokenizer, example):
    # TODO: Experiment with sampling to see if that does something.
    if isinstance(tokenizer, SentencePieceProcessor):
        # Directly specify that this is indeed input_ids, since it's not done automatically through HuggingFace wizard shit
        input_ids = tokenizer.encode_as_ids(example)

    # huggingface method of encoding with AutoTokenizer
    else:
        input_ids = tokenizer(example)

    return input_ids


#examples.remove_columns[args.select_input_string]

def make_test_dataloader():

    # Get the dataset
    tokenizer_args = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args)
    streamable_data = load_dataset("the_pile", name = "hacker_news", split = "train", streaming = True)

    # Run the tokenizer on each example
    map_tokenizer = lambda x: tokenize_example(tokenizer, x)
    tokenized_dataset = streamable_data.map(map_tokenizer, input_columns=args.select_input_string)

    # Turn the tokenized examples into a DataLoader for collosal AI to use
    dl = DataLoader(tokenized_dataset)
    #ds = DistributedSampler(dl)
    return dl


if __name__ == "__main__":
    test_dataloader = make_test_dataloader()
    stream_first_entry = next(iter(test_dataloader))
    print(stream_first_entry['input_ids']) 
    print(len(stream_first_entry['input_ids']))


# Needs to ahve
# - Shuffle (or db already shuffeled before download?)
# - Set type
# - Get length

