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


# tokenizer = GPT2Tokenizer.from_pretrained('code-tokenizer/',local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator = pipeline('text-generation', tokenizer=tokenizer, model='saved_dec4_gpt_b')

print(generator('print', max_length=5)[0]['generated_text'])
print(generator('print', max_length=5,num_beams = 5)[0]['generated_text'])
print(generator('print' , max_length=5 , do_sample=True,temperature = 0.7)[0]['generated_text'])

print(generator('for i in ', max_length=5)[0]['generated_text'])
print(generator('for i in ', max_length=5,num_beams = 5)[0]['generated_text'])
print(generator('for i in ' , max_length=5 , do_sample=True,temperature = 0.7)[0]['generated_text'])

print(generator('import ', max_length=5)[0]['generated_text'])
print(generator('import ', max_length=5,num_beams = 5)[0]['generated_text'])
print(generator('import ' , max_length=5 , do_sample=True,temperature = 0.7)[0]['generated_text'])