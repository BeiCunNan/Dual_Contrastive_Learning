import os
import json
import torch
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader


# 数据加载读取+文本预处理+Bert模型处理
# 获得Bert输出后的结果


class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        # 如果是dualcl,就将label进行list操作
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl', 'nl'] else []
        sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        dataset = list()
        # 数据读取和预处理
        for data in raw_data:
            tokens = data['text'].lower().split(' ')  # 将文本都转化为小写字母并按照空格来分隔,但是保留了标点符号
            label_id = label_dict[data['label']]  # 获取数字形式的标签id
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset

    # 如果是ce和scl方法,直接获得某索引的文本tokens和标签id
    # 如果是dualcl方法, 将标签信息加入到文本前面
    # 因为我们使用了collate_fn方法,其输入值为__getitem__方法返回的数据,因此我们一定要重写该方法
    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


# batch为每个batch的文本+标签集合，如[(['numeric', 'entity', 'human', 'location', 'abbreviation', 'description', '[SEP]', 'what', 'was', 'simple', 'simon', 'fishing', 'for', 'in', 'his', 'mother', "'", 's', 'pail', '?'], 1)
# Tokenizer为具体的预训练模型,如PreTrainedTokenizerFast(name_or_path='bert-base-uncased', vocab_size=30522, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
def my_collate(batch, tokenizer, method, num_classes):
    # *[] 的作用是将[a,b,c]变成[a],[b],[c]实际的数据是(['numeric', 'entity', 'human', 'location', 'abbreviation', 'description', '[SEP]', 'what', 'was', 'simple', 'simon', 'fishing', 'for', 'in', 'his', 'mother', "'", 's', 'pail', '?'], 1) (['numeric', 'entity', 'human', 'location', 'abbreviation', 'description', '[SEP]', 'what', 'was', 'simple', 'simon', 'fishing', 'for', 'in', 'his', 'mother', "'", 's', 'pail', '?'], 1)
    # zip(*batch)是将文本都集合在一起,标签都集合在一起
    # map(list,zip(*batch))是将文本集合成一个list,将标签集合成一个list
    tokens, label_ids = map(list, zip(*batch))  # 将标签集和文本集分离出来
    # tokens 为标签+文本，如['abbreviation', 'numeric', 'entity', 'human', 'location', 'description', '[SEP]', 'what', 'can', 'be', 'done', 'about', 'snoring', '?']
    # text_ids 为tokens化后的结果,包含input_ids,token_type_ids,attention_mask
    # 使用bert对文本实现序列化
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    # Dualcl中所有position_ids的内容为[0*类个数+0到结尾]如'position_ids': tensor([[ 0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
    #          12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1) - num_classes)
        text_ids['position_ids'] = positions
    return text_ids, torch.tensor(label_ids)


def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers):
    #  1 获取数据
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    else:
        raise ValueError('unknown dataset')

    # 2 预处理标签和文本,将其放到Dataset（类似于数据库）中
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)

    # 3 使用DataLoader（类似于怎么使用数据库）
    # num_workers 为进程数量,pin_memory为锁业内存二者均是为了加快训练速度的
    # partial 为偏函数,先输入一个tokernizer函数。my_collate为token化数据,即将数据长度变成一样的。随后Dataloader根据如train_batch_size来划分一个个batch的数据,并将这些数据输入到该偏函数中。
    # collate_fn 是重点,决定怎么取batch的数据,取出的数据可能长短不一,这是不允许的,因此需要使用=后面的整体 来整理数据长度
    # 最后 collate_fn中的my_collate的返回值就是DataLoader的返回值
    collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict))
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn,
                                  pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn,
                                 pin_memory=True)
    return train_dataloader, test_dataloader
