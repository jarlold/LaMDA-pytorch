from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, default_data_collator
from .config.config import CFG

import copy
from itertools import chain
from typing import Union

from sentencepiece import SentencePieceProcessor
from torch.distributed import get_world_size
import torch
args = CFG

# TODO:
# - General optimization and cleaning
# - Concat tokens more than once if they still aren't long enough


# Used to store the "offcut" after cutting the input from the db into the seq_length form config
PREV_OFFCUT = torch.tensor([], dtype=torch.int32)

def prepare_example(tokenizer, example):
    global PREV_OFFCUT
    seq_length = args.tokenizer_seq_length

    # Start by tokenizing 
    if isinstance(tokenizer, SentencePieceProcessor):
        # Directly specify that this is indeed input_ids, since it's not done automatically through HuggingFace wizard shit
        input_dict = tokenizer.encode_as_ids(example)

    # huggingface method of encoding with AutoTokenizer
    else:
        input_dict = tokenizer(example)

    result = {}
    result['input_ids'] = []

    # Concatenate  TODO: faster ways to do this
    for i in input_dict['input_ids']:
        result['input_ids'].append(i)

    result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.int32)

    # Then we have to make sure they're actually the right sequence length
    comb = torch.cat((PREV_OFFCUT, result['input_ids'])) # since we do this AFTER tokenization we use 'input_ids' not args.select_input_string
    #comb = comb[0]
    PREV_OFFCUT= comb[args.tokenizer_seq_length:]

    input_ids = comb[:args.tokenizer_seq_length].reshape([512])
    result['input_ids'] = input_ids
    result["labels"] = input_ids
    return result

class WrapDataloader(DataLoader):
    def __len__(self):
        return self.manual_length

    def manually_set_length(self, length):
        self.manual_length = length * 0


def build_dataloaders():
    # Get the dataset
    tokenizer_args = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args)

    load_train_data = load_dataset(args.train_dataset_path, name = args.train_dataset_name, split = args.choose_train_split, streaming = True)
    load_test_data = load_dataset(args.train_dataset_path, name = args.train_dataset_name, split = args.choose_eval_split, streaming = True)

    # Run the tokenizer on each example
    map_tokenizer = lambda x: prepare_example(tokenizer, x)
    tokenized_train_data = load_train_data.map(map_tokenizer, input_columns=args.select_input_string)
    tokenized_test_data = load_test_data.map(map_tokenizer, input_columns=args.select_input_string)

    # If we do not remove these, colossal AI will assume they are input tokens. This is because colossal AI was made by evil
    # goblin people who hate me.
    tokenized_train_data = tokenized_train_data.remove_columns(['text', 'meta'])
    tokenized_train_data = tokenized_test_data.remove_columns(['text', 'meta'])

    # And now we'll need to add in some sort of length function
    # Since streaming doesn't let us know this, we'll just put it in the config
    train_dl = WrapDataloader(tokenized_train_data)
    test_dl = WrapDataloader(tokenized_test_data)

    # Set the length
    train_dl.manually_set_length(args.train_len_if_stream)
    test_dl.manually_set_length(args.eval_len_if_stream)

    # Put our little sequence length fix over it
    return train_dl, test_dl

