# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


#import FunsdDataset, LayoutlmConfig, LayoutlmForTokenClassification
from layoutlm import DocVQADataset, LayoutlmConfig, LayoutlmForTokenClassification


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, LayoutlmConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
}


# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)


# def collate_fn(data):
#     batch = [i for i in zip(*data)]
#     for i in range(len(batch)):
#         if i < len(batch) - 2:
#             batch[i] = torch.stack(batch[i], 0)
#     return tuple(batch)


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(  # noqa C901
    args, train_dataset, model, tokenizer, labels, pad_token_label_id
):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir="runs/" + os.path.basename(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=None,
    )

    t_total = len(train_dataloader)
    print("Total data length: %d" % t_total)
           

def main():  # noqa C901
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    #parser.add_argument(
    #     "--do_train", action="store_true", help="Whether to run training."
    # )
    # parser.add_argument(
    #     "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    # )
    # parser.add_argument(
    #     "--do_predict",
    #     action="store_true",
    #     help="Whether to run predictions on the test set.",
    # )
    # parser.add_argument(
    #     "--evaluate_during_training",
    #     action="store_true",
    #     help="Whether to run evaluation during training at each logging step.",
    # )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    # parser.add_argument(
    #     "--per_gpu_train_batch_size",
    #     default=8,
    #     type=int,
    #     help="Batch size per GPU/CPU for training.",
    # )
    # parser.add_argument(
    #     "--per_gpu_eval_batch_size",
    #     default=8,
    #     type=int,
    #     help="Batch size per GPU/CPU for evaluation.",
    # )
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    # parser.add_argument(
    #     "--learning_rate",
    #     default=5e-5,
    #     type=float,
    #     help="The initial learning rate for Adam.",
    # )
    # parser.add_argument(
    #     "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    # )
    # parser.add_argument(
    #     "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    # )
    # parser.add_argument(
    #     "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    # )
    # parser.add_argument(
    #     "--num_train_epochs",
    #     default=3.0,
    #     type=float,
    #     help="Total number of training epochs to perform.",
    # )
    # parser.add_argument(
    #     "--max_steps",
    #     default=-1,
    #     type=int,
    #     help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    # )
    # parser.add_argument(
    #     "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    # )

    # parser.add_argument(
    #     "--logging_steps", type=int, default=50, help="Log every X updates steps."
    # )
    # parser.add_argument(
    #     "--save_steps",
    #     type=int,
    #     default=50,
    #     help="Save checkpoint every X updates steps.",
    # )
    # parser.add_argument(
    #     "--eval_all_checkpoints",
    #     action="store_true",
    #     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    # )
    # parser.add_argument(
    #     "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    # )
    # parser.add_argument(
    #     "--overwrite_output_dir",
    #     action="store_true",
    #     help="Overwrite the content of the output directory",
    # )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    # parser.add_argument(
    #     "--seed", type=int, default=42, help="random seed for initialization"
    # )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    # parser.add_argument(
    #     "--server_ip", type=str, default="", help="For distant debugging."
    # )
    # parser.add_argument(
    #     "--server_port", type=str, default="", help="For distant debugging."
    # )
    args = parser.parse_args()

    # if (
    #     os.path.exists(args.output_dir)
    #     and os.listdir(args.output_dir)
    #     and args.do_train
    # ):
    #     if not args.overwrite_output_dir:
    #         raise ValueError(
    #             "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #                 args.output_dir
    #             )
    #         )
    #     else:
    #         if args.local_rank in [-1, 0]:
    #             shutil.rmtree(args.output_dir)

    # if not os.path.exists(args.output_dir) and (args.do_eval or args.do_predict):
    #     raise ValueError(
    #         "Output directory ({}) does not exist. Please train and save the model before inference stage.".format(
    #             args.output_dir
    #         )
    #     )

    # if (
    #     not os.path.exists(args.output_dir)
    #     and args.do_train
    #     and args.local_rank in [-1, 0]
    # ):
    #     os.makedirs(args.output_dir)

    # # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd

    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(
    #         address=(args.server_ip, args.server_port), redirect_output=True
    #     )
    #     ptvsd.wait_for_attach()

    # # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device(
    #         "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    #     )
    #     args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl")
    #     args.n_gpu = 1
    # args.device = device

    # # Setup logging
    # logging.basicConfig(
    #     filename=os.path.join(args.output_dir, "train.log")
    #     if args.local_rank in [-1, 0]
    #     else None,
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    # )
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     args.local_rank,
    #     device,
    #     args.n_gpu,
    #     bool(args.local_rank != -1),
    #     args.fp16,
    # )

    # # Set seed
    # set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    # # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(
    #     args.config_name if args.config_name else args.model_name_or_path,
    #     num_labels=num_labels,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    
    train_dataset = DocVQADataset(
        args, tokenizer, labels, pad_token_label_id, mode="test"
    )
    print('Dataloaded! ')
    print("Length of dataset", len(train_dataset))

if __name__ == "__main__":
    main()

