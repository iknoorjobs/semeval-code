!/usr/bin/env python
# coding: utf-8

import math,re
from collections import Counter
import numpy as np
import json
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
from collections import Counter
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import sklearn


train1 = json.load(open("semevalEnrichedTrainTest/train.json"))
train2 = json.load(open("semevalEnrichedTrainTest/test.json"))

# add augumented data here
aug1 = json.load(open("aug1.json"))  
aug2 = json.load(open("aug2.json"))  
aug3 = json.load(open("aug3.json"))  


train = train1 + train2 + aug1 + aug2

semevalData = json.load(open("semevalOldTrainTest/semevalData.json"))
for i in range(len(train)):
    pair_id = train[i]['pair_id']
    train[i]["url1_publish_date"] = semevalData[pair_id.split("_")[0]]['publish_date']
    train[i]["url2_publish_date"] = semevalData[pair_id.split("_")[1]]['publish_date'] 

    
train = train + aug3 #+ aug4

def counter_cosine_similarity(c1, c2):
    if not c1 or not c2: return float(0)
    Entities = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in Entities)
    MagnitudeA = math.sqrt(sum(c1.get(k, 0)**2 for k in Entities))
    MagnitudeB = math.sqrt(sum(c2.get(k, 0)**2 for k in Entities))
    return dotprod / (MagnitudeA * MagnitudeB)


features = []

true=[]
pred=[]
for item in train:
    gpe1 = item['url1_ner']['GPE'] if 'GPE' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['LOC'] if 'LOC' in item['url1_ner'] else []
    gpe = gpe1+gpe2
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1 = item['url2_ner']['GPE'] if 'GPE' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['LOC'] if 'LOC' in item['url2_ner'] else []
    gpe = gpe1+gpe2
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))

    
features.append(pred)


true=[]
pred=[]
for item in train:
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url1_ner']['PERSON'] if 'PERSON' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['ORG'] if 'ORG' in item['url1_ner'] else []
    gpe3 = item['url1_ner']['FAC'] if 'FAC' in item['url1_ner'] else []
    gpe4 = item['url1_ner']['EVENT'] if 'EVENT' in item['url1_ner'] else []
    gpe5 = item['url1_ner']['NORP'] if 'NORP' in item['url1_ner'] else []
    gpe6 = item['url1_ner']['PRODUCT'] if 'PRODUCT' in item['url1_ner'] else []
    gpe7 = item['url1_ner']['WORK_OF_ART'] if 'WORK_OF_ART' in item['url1_ner'] else []
    gpe = gpe1+gpe2+gpe3+gpe4+gpe5+gpe6+gpe7
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url2_ner']['PERSON'] if 'PERSON' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['ORG'] if 'ORG' in item['url2_ner'] else []
    gpe3 = item['url2_ner']['FAC'] if 'FAC' in item['url2_ner'] else []
    gpe4 = item['url2_ner']['EVENT'] if 'EVENT' in item['url2_ner'] else []
    gpe5 = item['url2_ner']['NORP'] if 'NORP' in item['url2_ner'] else []
    gpe6 = item['url2_ner']['PRODUCT'] if 'PRODUCT' in item['url2_ner'] else []
    gpe7 = item['url2_ner']['WORK_OF_ART'] if 'WORK_OF_ART' in item['url2_ner'] else []
    gpe = gpe1+gpe2+gpe3+gpe4+gpe5+gpe6+gpe7
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))
    
    
features.append(pred)


true=[]
pred=[]
for item in train:
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url1_ner']['DATE'] if 'DATE' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['TIME'] if 'TIME' in item['url1_ner'] else []
    gpe = gpe1+gpe2
    if 'url1_publish_date' in item and item['url1_publish_date']: gpe.append(item['url1_publish_date'][:-14].strip())
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url2_ner']['DATE'] if 'DATE' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['TIME'] if 'TIME' in item['url2_ner'] else []
    gpe = gpe1+gpe2
    if 'url2_publish_date' in item and item['url2_publish_date']: gpe.append(item['url2_publish_date'][:-14].strip())
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))
    
    
features.append(pred)

true=[]
pred=[]
for item in train:
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url1_ner']['ORDINAL'] if 'ORDINAL' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['CARDINAL'] if 'CARDINAL' in item['url1_ner'] else []
    gpe3 = item['url1_ner']['QUANTITY'] if 'QUANTITY' in item['url1_ner'] else []
    gpe = gpe1+gpe2+gpe3
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url2_ner']['ORDINAL'] if 'ORDINAL' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['CARDINAL'] if 'CARDINAL' in item['url2_ner'] else []
    gpe3 = item['url2_ner']['QUANTITY'] if 'QUANTITY' in item['url2_ner'] else []
    gpe = gpe1+gpe2+gpe3
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))
    
    
features.append(pred)




device="cuda:3"
# Add trained Simaese Transformer model here using src/train1.py
modelName = ""
model = SentenceTransformer(modelName, device=device)

train_samples = []

for item in train:
    score = item['OverallNorm']
    if "url1_title" in item:
        url1_text = item['url1_title'] + " " + item['url1_text']
    else:
        url1_text = item['url1_text']
    if "url2_title" in item:
        url2_text = item['url2_title'] + " " + item['url2_text']
    else:
        url2_text = item['url2_text']
#     url1_text, url2_text = datasetFormat(item)
    inp_example = InputExample(texts=[url1_text, url2_text], label=score)
    train_samples.append(inp_example)
    

train_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(train_samples, name='sts-test')
embeddings1 = model.encode(train_evaluator.sentences1, convert_to_numpy=True, batch_size=512)
embeddings2 = model.encode(train_evaluator.sentences2, convert_to_numpy=True, batch_size=512)
distances = 1 - (paired_cosine_distances(embeddings1, embeddings2))
newman = sklearn.preprocessing.minmax_scale(distances, feature_range=(0, 1), axis=0)
features.append(newman)


y = []
for i in range(len(train)):
    y.append(train[i]['OverallNorm'])


features = np.asarray(features)
X = features.T
y = np.asarray(y)


from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=49, max_iter=4000 , hidden_layer_sizes=32, activation="logistic",
                   learning_rate_init=0.001, solver="adam", alpha=0.01, learning_rate="adaptive").fit(X, y)





# ****************** Add test data here ***************************
train = json.load(open("testDataWithTranslationNer.json"))

featuresTest = []

true=[]
pred=[]
for item in train:
    gpe1 = item['url1_ner']['GPE'] if 'GPE' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['LOC'] if 'LOC' in item['url1_ner'] else []
    gpe = gpe1+gpe2
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1 = item['url2_ner']['GPE'] if 'GPE' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['LOC'] if 'LOC' in item['url2_ner'] else []
    gpe = gpe1+gpe2
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))

    
featuresTest.append(pred)


true=[]
pred=[]
for item in train:
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url1_ner']['PERSON'] if 'PERSON' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['ORG'] if 'ORG' in item['url1_ner'] else []
    gpe3 = item['url1_ner']['FAC'] if 'FAC' in item['url1_ner'] else []
    gpe4 = item['url1_ner']['EVENT'] if 'EVENT' in item['url1_ner'] else []
    gpe5 = item['url1_ner']['NORP'] if 'NORP' in item['url1_ner'] else []
    gpe6 = item['url1_ner']['PRODUCT'] if 'PRODUCT' in item['url1_ner'] else []
    gpe7 = item['url1_ner']['WORK_OF_ART'] if 'WORK_OF_ART' in item['url1_ner'] else []
    gpe = gpe1+gpe2+gpe3+gpe4+gpe5+gpe6+gpe7
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url2_ner']['PERSON'] if 'PERSON' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['ORG'] if 'ORG' in item['url2_ner'] else []
    gpe3 = item['url2_ner']['FAC'] if 'FAC' in item['url2_ner'] else []
    gpe4 = item['url2_ner']['EVENT'] if 'EVENT' in item['url2_ner'] else []
    gpe5 = item['url2_ner']['NORP'] if 'NORP' in item['url2_ner'] else []
    gpe6 = item['url2_ner']['PRODUCT'] if 'PRODUCT' in item['url2_ner'] else []
    gpe7 = item['url2_ner']['WORK_OF_ART'] if 'WORK_OF_ART' in item['url2_ner'] else []
    gpe = gpe1+gpe2+gpe3+gpe4+gpe5+gpe6+gpe7
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))
    
    
featuresTest.append(pred)


true=[]
pred=[]
for item in train:
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url1_ner']['DATE'] if 'DATE' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['TIME'] if 'TIME' in item['url1_ner'] else []
    gpe = gpe1+gpe2
    if item['url1_publish_date']: gpe.append(item['url1_publish_date'][:-14].strip())
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url2_ner']['DATE'] if 'DATE' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['TIME'] if 'TIME' in item['url2_ner'] else []
    gpe = gpe1+gpe2
    if item['url2_publish_date']: gpe.append(item['url2_publish_date'][:-14].strip())
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))
    
    
featuresTest.append(pred)

true=[]
pred=[]
for item in train:
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url1_ner']['ORDINAL'] if 'ORDINAL' in item['url1_ner'] else []
    gpe2 = item['url1_ner']['CARDINAL'] if 'CARDINAL' in item['url1_ner'] else []
    gpe3 = item['url1_ner']['QUANTITY'] if 'QUANTITY' in item['url1_ner'] else []
    gpe = gpe1+gpe2+gpe3
    gpe = [it.lower().strip() for it in gpe]
    d1 = Counter(gpe)
    
    gpe1,gpe2,gpe3 = [],[],[]
    gpe1 = item['url2_ner']['ORDINAL'] if 'ORDINAL' in item['url2_ner'] else []
    gpe2 = item['url2_ner']['CARDINAL'] if 'CARDINAL' in item['url2_ner'] else []
    gpe3 = item['url2_ner']['QUANTITY'] if 'QUANTITY' in item['url2_ner'] else []
    gpe = gpe1+gpe2+gpe3
    gpe = [it.lower().strip() for it in gpe]
    d2 = Counter(gpe)
    
    pred.append(counter_cosine_similarity(d1,d2))
    
featuresTest.append(pred)


semeeval = json.load(open("semevalEvaluationTestSet/testDataWithTranslationNer.json"))
semeeval_samples = []
for item in semeeval:
    url1_text = item['url1_title'] + " " + item['url1_text']
    url2_text = item['url2_title'] + " " + item['url2_text']
#     url1_text, url2_text = datasetFormat(item)
    inp_example = InputExample(texts=[url1_text, url2_text])
    semeeval_samples.append(inp_example)

semeeval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(semeeval_samples, name='sts-test')
embeddings1 = model.encode(semeeval_evaluator.sentences1, convert_to_numpy=True, batch_size=512)
embeddings2 = model.encode(semeeval_evaluator.sentences2, convert_to_numpy=True, batch_size=512)


distances = 1 - (paired_cosine_distances(embeddings1, embeddings2))
newman = sklearn.preprocessing.minmax_scale(distances, feature_range=(0, 1), axis=0)
featuresTest.append(list(newman))


featuresTest = np.asarray(featuresTest)
X_test = featuresTest.T


pred = regr.predict(X_test)
newman = sklearn.preprocessing.minmax_scale(pred, feature_range=(0, 1), axis=0)
predictions = 4.0 - (3*newman)


df = pd.DataFrame(predictions)
pair_id = []
for i in range(len(semeeval)): pair_id.append(semeeval[i]['pair_id'])
df['pair_id'] = pair_id
df.columns = [["Overall", "pair_id"]]
df = df[['pair_id','Overall']]



x = 1 # ******* SET RUN Number *******
df.to_csv("semevalEvaluationTestSet/finalSubmissions/gatenlpi"+str(x)+".csv", index=False)






