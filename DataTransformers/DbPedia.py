import csv
import json
import re

import numpy as np
import pandas as pd

PATH = "E:\data\dbpedia_csv\\train.csv"
dict = {"1": "company", "2": "educationalInstitution", "3": "artist", "4": "athlete", "5": "officeholder",
        "6": "meanoftransportation", "7": "building", "8": "naturalplace", "9": "village",
        "10": "animal", "11": "plant", "12": "album", "13": "film", "14": "writtenwork"}

df = pd.read_csv(PATH, encoding="iso-8859-1")

result = []
for index in range(len(df)):
    k=type(df["text"][index])
    if(k!=type("nihao")):
        item = {
            "text": df["text"][index-100].strip('.').strip(),
            "title": df["title"][index-100],
            "label": dict[str(df["label"][index-100])]
        }
    else:
        item = {
            "text": df["text"][index].strip('.').strip(),
            "title": df["title"][index],
            "label": dict[str(df["label"][index])]
        }
    result.append(item)

with open('../data/DP_Train.json', 'w') as w:
    json.dump(result, w)
