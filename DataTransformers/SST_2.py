import json
import re

PATH = "E:\data\SST_5\\test.txt"
dict={"0":"one","1":"two","2":"three","3":"four","4":"five"}

with open(PATH, 'r', encoding="utf-8") as f:
    result=[]
    for line in f:
        line = ''.join(line).strip().lstrip('\t')
        l = line.split("\t")
        label = dict[l[0]]
        text = l[1].strip('.').strip()
        item={
            "text":text,
            "label":label
        }
        result.append(item)

with open('../data/SST5_Test.json', 'w') as w:
    json.dump(result,w)
