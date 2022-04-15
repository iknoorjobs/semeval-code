import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import defaultdict
import json

data = json.load(open("trainDataWithTranslation.json"))
nlp = spacy.load('en_core_web_trf', disable = ['tagger', 'parser',"lemmatizer"])

for i in range(len(data)):
    print(i)
    doc1 = nlp(data[i]['url1_title_en'] + " " + data[i]['url1_text_en'])
    meta1 = defaultdict(list)
    for ent in doc1.ents:
        meta1[ent.label_].append(ent.text)
    data[i]['url1_ner'] = meta1
    doc2 = nlp(data[i]['url2_title_en'] + " " + data[i]['url2_text_en'])
    meta2 = defaultdict(list)
    for ent in doc2.ents:
        meta2[ent.label_].append(ent.text)
    data[i]['url2_ner'] = meta2


with open('trainDataWithTranslationNer.json', 'w') as fp:
    json.dump(data, fp)

