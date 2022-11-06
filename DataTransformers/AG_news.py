import csv
import json
import re

import numpy as np
import pandas as pd

PATH = "E:\data\\ag_news_csv\\train.csv"
dict = {"1": "world", "2": "sports", "3": "business", "4": "technology"}

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

with open('../data/AG_Train.json', 'w') as w:
    json.dump(result, w)
