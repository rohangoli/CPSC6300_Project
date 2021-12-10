import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, TimeDistributed, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pickle
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

import unicodedata
import re
import os
import time
import shutil
import requests
import tarfile
import glob

import argparse
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from io import BytesIO
import json

import pandas as pd
import numpy as np
import string, os

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel,GPT2LMHeadModel
from transformers import TextDataset,TrainingArguments,Trainer,pipeline,DataCollatorForLanguageModeling
import torch

print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)

with open("train_sent.txt", "r") as fp: 
    train_sent = fp.read().splitlines() 
with open("test_sent.txt", "r") as fp:
    test_sent = fp.read().splitlines() 
with open("full_corpus.txt", "r") as fp:
    full_corpus = fp.read().splitlines() 

tokenizer = GPT2Tokenizer.from_pretrained('code-tokenizer-scratch/',local_files_only=True)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = TFGPT2LMHeadModel.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('saved_dec4_gpt_c', local_files_only=True)

use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3], dim=0)
if use_cuda:
    model = model.cuda()

print('vocabulary size: %d, max sequence length: %d' % (tokenizer.vocab_size, tokenizer.model_max_length))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='train_sent.txt',
    overwrite_cache=True,
    block_size=24)

test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='test_sent.txt',
    overwrite_cache=True,
    block_size=24)

training_args = TrainingArguments(
    output_dir = 'dec4_gpt', 
    overwrite_output_dir = True, 
    per_device_train_batch_size = 128, 
    per_device_eval_batch_size = 128, 
    learning_rate = 5e-4, 
    save_steps=1000,
    logging_steps=3000,
    save_total_limit=2,
    num_train_epochs = 3,
)

# Initializing the trainer class object that will do the training
# here the data collator will generate the batch of size 64 of train and test data
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator=data_collator,
    train_dataset = train_dataset,
    eval_dataset = test_dataset
)

trainer.train()

trainer.save_model('./saved_dec4_gpt_c')

# Evaluating on Test data
trainer.evaluate(test_dataset)