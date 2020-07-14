#!/usr/bin/env python

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 University of Stuttgart (Pavel Denisov)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Generate Transformer embeddings for Kaldi text file."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import tqdm
import torch

import numpy as np

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertModel, BertTokenizer,
                                  XLNetConfig, XLNetModel,
                                  XLNetTokenizer, RobertaConfig,
                                  RobertaModel, RobertaTokenizer)

from sentence_transformers import SentenceTransformer 

from espnet.utils.cli_writers import file_writer_helper

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetModel, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'sbert': (None, None, None)
}

class InputExample(object):

    def __init__(self, unique_id, text):
        self.unique_id = unique_id
        self.text = text

def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = ''
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            unique_id = line[:line.index(' ')]
            text = line[line.index(' ')+1:]
            examples.append(
                InputExample(unique_id=unique_id, text=text))
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
    # length or only pop from context
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            logger.info('Attention! you are removing from token_b (swag task is ok). '
                        'If you are training ARC and RACE (you are poping question + options), '
                        'you need to try to use a bigger max seq length!')
            tokens_b.pop()

def convert_examples_to_features(examples, writer, max_seq_length,
                                 tokenizer, transformer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=0,
                                 mask_padding_with_zero=True,
                                 device='cpu'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text)
        tokens_b = None
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, [], max_seq_length - special_tokens_count)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        writer[example.unique_id] = torch.mean(transformer(
            input_ids=torch.tensor([input_ids], dtype=torch.long).to(device),
            attention_mask=torch.tensor([input_mask], dtype=torch.long).to(device),
            token_type_ids=torch.tensor([segment_ids], dtype=torch.long).to(device)
            )[0], 1).detach().cpu().numpy()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_convert_examples(args, tokenizer, transformer):
    examples = read_examples(args.input_file)

    logger.info("Training number: %s", str(len(examples)))
    with file_writer_helper(args.wspecifier,
                              filetype=args.filetype,
                              compress=args.compress,
                              compression_method=args.compression_method
                              ) as writer:
        convert_examples_to_features(examples, writer, args.max_seq_length, tokenizer, transformer,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
            device=args.device)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for output. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--compress', type=bool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')

    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO)

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()

    if args.model_type == 'sbert':
        transformer = SentenceTransformer(args.model_name_or_path)
        examples = read_examples(args.input_file)
        embeddings = transformer.encode([e.text for e in examples])

        with file_writer_helper(args.wspecifier,
                                  filetype=args.filetype,
                                  compress=args.compress,
                                  compression_method=args.compression_method
                                  ) as writer:
            for i in range(len(examples)):
                writer[examples[i].unique_id] = embeddings[i]
    else:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        transformer = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

        transformer.eval()
        transformer.to(args.device)

        with torch.no_grad():
            load_and_convert_examples(args, tokenizer, transformer)

    logger.info('Done converting {} to {}'.format(args.input_file, args.wspecifier))


if __name__ == "__main__":
    main()
