import csv
import json
import re

import numpy as np
import pandas as pd

PATH = "E:\data\yelp_review_full_csv\\train.csv"
dict={"1":"one","2":"two","3":"three","4":"four","5":"five"}

df = pd.read_csv(PATH)

result = []
for index in range(len(df)):
    item = {
        "text": df["text"][index].strip('.').strip(),
        "label": dict[str(df["label"][index])]
    }
    result.append(item)

with open('../data/Yelp5_Train.json', 'w') as w:
    json.dump(result, w)


