import sys
import logging
import os
import argparse
from transformers import BartTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers import BartPretrainedModel, BartConfig
from model.modeling_bart import LMEDRModel
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
import math
from pprint import pformat
from build_data_PersonaChat import create_data, build_dataloader, build_infer_dataset

checkpoint = torch.load('/home/yiwang/chatbot-LMEDR/persona_original/checkpoint_mymodel_3126.pt', map_location='cpu')
config = BartConfig(vocab_size=50269,num_labels=1)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
add_special_tokens = {'additional_special_tokens': ['<query>', '<response>', '<latent>', '<persona>']}
num_added_toks = tokenizer.add_special_tokens(add_special_tokens)
# logger.info('We have added {} tokens'.format(num_added_toks))
model = LMEDRModel.from_pretrained("facebook/bart-large", num_labels=1,
                                    num_token=len(tokenizer),
                                    num_latent=10, num_latent2=10)
model.resize_token_embeddings(len(tokenizer))
model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids('<response>')
model.config.forced_bos_token_id = None
model.load_state_dict(checkpoint)
print(model.state_dict().keys())
torch.save(model.state_dict(), '/home/yiwang/chatbot-LMEDR/persona_original/checkpoint/pytorch_model.bin')

# torch.save(torch.randn(10, 10), '/home/yiwang/chatbot-LMEDR/persona_original/checkpoint/dummy.pt')

