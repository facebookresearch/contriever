# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
import torch
import logging
import json
import numpy as np
import os

import transformers

import src.slurm
import src.contriever
import src.beir_utils
import src.utils as utils
import src.dist_utils as dist_utils

logger = logging.getLogger(__name__)


def main(args):

    src.slurm.init_distributed_mode(args)
    src.slurm.init_signal_handler()

    os.makedirs(args.output_dir, exist_ok=True)

    logger = utils.init_logger(args)
    logger.info(f"Loading model from {args.model_name_or_path}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = src.contriever.Contriever.from_pretrained(args.model_name_or_path)

    model = model.cuda()

    logger.info("Start indexing")

    ndcg, _map, recall, precision, mrr, recall_cap, hole = src.beir_utils.evaluate_model(
        query_encoder=model, 
        doc_encoder=model,
        tokenizer=tokenizer, 
        dataset=args.dataset,
        batch_size=args.per_gpu_batch_size,
        norm_query=args.norm_query,
        norm_doc=args.norm_doc,
        is_main=dist_utils.is_main(),
        split='dev' if args.dataset=='msmarco' else 'test',
        metric=args.metric,
        beir_data_path=args.beir_data_path,
    )

    if dist_utils.is_main():
        logger.info(args.dataset + ' ' + str(ndcg))
        logger.info(args.dataset + ' ' + str(recall))
        logger.info(args.dataset + ' ' + str(precision))
        logger.info(args.dataset + ' ' + str(mrr))
        logger.info(args.dataset + ' ' + str(recall_cap))
        logger.info(args.dataset + ' ' + str(hole))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_data_path", type=str, default="BEIR/datasets", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--metric", type=str, default="dot", 
        help="Metric used to compute similarity between two embeddings")
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")


    args, _ = parser.parse_known_args()
    main(args)
    
