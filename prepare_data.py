import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser


logger = logging.getLogger(__name__)


class QGDataset(torch.utils.data.Dataset):
    def __init__(self, examples, dtype, model_type, max_source_length=512, max_target_length=32):
        self.examples = examples[:1000]
        self.dtype = dtype

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type

        if model_type == 't5':
            self.sep_token = '<sep>'
        elif model_type == 'bart':
            self.sep_token = '<sep>'
        else:
            self.sep_token = '[SEP]'

    @staticmethod
    def _add_eos(example):
        example['graph_seq'] += ' </s>'
        example['q_toks'] += ' </s>'
        return example

    def preprocess(self, tokenizer):
        for i, example in tqdm(enumerate(self), total=len(self)):
            example = self._add_eos(example)

            source_encoding = tokenizer.encode_plus(
                example['graph_seq'],
                max_length=self.max_source_length,
                padding='max_length',
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt'
            )
            target_encoding = tokenizer.encode_plus(
                example['q_toks'],
                max_length=self.max_target_length,
                padding='max_length',
                pad_to_max_length=True,
                truncation=True,
                return_tensors='pt'
            )

            encodings = {
                'source_ids': source_encoding['input_ids'],
                'target_ids': target_encoding['input_ids'],
                'attention_mask': source_encoding['attention_mask'],
            }

            example.update(encodings)
            self.examples[i] = example

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help":
                      "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"},
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default='~/Desktop/aqa/data/squad',
        metadata={'help': 'data directory for train and validation csv files.'}
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    qg_format: Optional[str] = field(
        default='highlight_qg_format',
        metadata={'help': "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={'help': 'Max input length for the source text'},
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={'help': 'Max input length for the target text'},
    )


def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
    else:
        tokenizer = T5Tokenizer.from_pretrained('bart-base')
    
    tokenizer.add_tokens(['<sep>', '<hl>', '<s>', '<o>', '<v>'])
    
    train_dataset = pd.read_csv(os.path.join(data_args.dataset_path, 'dataset_train.csv'))
    valid_dataset = pd.read_csv(os.path.join(data_args.dataset_path, 'dataset_validation.csv'))

    train_dataset.dropna(inplace=True)
    valid_dataset.dropna(inplace=True)

    min_qrecall = 0.33
    train_dataset = train_dataset[train_dataset['q_tok_recall'] >= min_qrecall]
    valid_dataset = valid_dataset[valid_dataset['q_tok_recall'] >= min_qrecall]
    train_records = train_dataset.to_dict('records')
    valid_records = valid_dataset.to_dict('records')

    train_dataset = QGDataset(
        train_records,  dtype='train', model_type=data_args.model_type, max_target_length=data_args.max_target_length,
        max_source_length=data_args.max_source_length
    )
    valid_dataset = QGDataset(
        valid_records, dtype='validation', model_type=data_args.model_type,
        max_target_length=data_args.max_target_length, max_source_length=data_args.max_source_length
    )

    train_dataset.preprocess(tokenizer)
    valid_dataset.preprocess(tokenizer)

    if data_args.train_file_name is None:
        train_file_name = f'train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt'
        train_path = os.path.join('data', train_file_name)

        valid_file_name = f'valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt'
        valid_path = os.path.join('data', valid_file_name)
    else:
        train_path = os.path.join('data', data_args.train_file_name)
        valid_path = os.path.join('data', data_args.valid_file_name)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()
