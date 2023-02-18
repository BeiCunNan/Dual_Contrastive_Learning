import csv
import json
import re

import numpy as np
import pandas as pd

PATH = "E:\data\mr\\rt-polarity.neg"
dict={"0":"positive","1":"negative"}

result = []
with open(PATH,'r',encoding="utf-8", errors='ignore') as f:
    while(True):
        line=f.readline()
        if not line:
            break
        item = {
            "text": line,
            "label": "positive"
        }
        result.append(item)


with open('../data/MR.json', 'w',encoding="utf-8", errors='ignore') as w:
    json.dump(result, w)


