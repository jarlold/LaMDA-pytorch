from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, default_data_collator
from config.config import CFG

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
PREV_OFFCUT = torch.tensor([])


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

def make_streamable_dataloader():
    # Get the dataset
    tokenizer_args = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args)

    load_train_data = load_dataset(args.train_dataset_path, name = args.train_dataset_name, split = args.choose_train_split, streaming = True)
    load_test= load_dataset(args.train_dataset_path, name = args.train_dataset_name, split = args.choose_test_split, streaming = True)

    # Run the tokenizer on each example
    map_tokenizer = lambda x: prepare_example(tokenizer, x)
    tokenized_train_data = load_train_data.map(map_tokenizer, input_columns=args.select_input_string)
    tokenized_test_data = load_test_data.map(map_tokenizer, input_columns=args.select_input_string)

    # Turn the tokenized examples into a DataLoader for collosal AI to use
    train_dl = DataLoader(tokenized_train_data)
    test_dl= DataLoader(tokenized_test_data)

    # Put our little sequence length fix over it
    return train_dl, test_dl




if __name__ == "__main__":
    streamable_data = make_test_dataloader()

    #test_dataloader = make_test_dataloader()
    print("getting first entry")
    stream_first_entry = next(iter(streamable_data))
    print(stream_first_entry['input_ids']) 
    print('next shape', stream_first_entry['input_ids'].shape)

    print('nxt')
    stream_first_entry = next(iter(streamable_data))
    print(stream_first_entry['input_ids']) 
    print('next shape', stream_first_entry['input_ids'].shape)

