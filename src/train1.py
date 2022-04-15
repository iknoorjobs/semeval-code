# The code has been adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark_continue_training.py

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import copy
import json
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
from collections import Counter


train = json.load(open("semevalEnrichedTrainTest/train.json"))
test = json.load(open("semevalEnrichedTrainTest/test.json"))

train_samples = []

for item in train:
    score = item['OverallNorm']
    url1_text = item['url1_title'] + " " + item['url1_text']
    url2_text = item['url2_title'] + " " + item['url2_text']
#     url1_text, url2_text = datasetFormat(item)
    inp_example = InputExample(texts=[url1_text, url2_text], label=score)
    train_samples.append(inp_example)


#You can specify any huggingface/transformers pre-trained model here, for example,
# paraphrase-multilingual-mpnet-base-v2
# LaBSE
# distiluse-base-multilingual-cased-v1
model_name = "distiluse-base-multilingual-cased-v2"

# Read the dataset
train_batch_size = 8
model_save_path = 'output/trainBiEncoderMMTweets-'+model_name.replace('/','-')+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# model_save_path = 'output/crossLingualModels/trainBiEncoderMMTweets-'+model_name.replace('/','-')+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device="cuda:3"

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name, device=device)
model.max_seq_length=512

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

test_samples = []
for item in test:
    score = item['OverallNorm']
    
    url1_text = item['url1_title'] + " " + item['url1_text']
    url2_text = item['url2_title'] + " " + item['url2_text']
#     url1_text, url2_text = datasetFormat(item)
    
    inp_example = InputExample(texts=[url1_text, url2_text], label=score)
    test_samples.append(inp_example)

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=4,
          evaluator=dev_evaluator,
          output_path=model_save_path,
          optimizer_params =  {'lr': 2e-5},
          use_amp=True)


model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
print(test_evaluator(model, output_path=model_save_path))
