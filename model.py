import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, base_model, num_classes, method):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)  # 防止过拟合,每次迭代训练的时候会有50%的数据量会被丢弃
        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        # Bert operations1
        # print('ooooooooo',inputs['input_ids'].shape)

        # Bert 一共有四种输出
        # last_hidden_state最后一层隐藏层所有单词的token值
        # pooler_output 序列第一个token的最后一层隐层状态,也就是CLS
        # hidden_states 不重要！
        # attentions 不重要！
        # 因此获取CLS的方法有两个,pooler_output or last_hidden_state[0,:]
        raw_outputs = self.base_model(**inputs)
        hiddens = raw_outputs.last_hidden_state  # last_hidden_state(batch_size,sequence_length,hidden_size),sequence_length为句子长度,hidden_size统一为768,输出如[16,15,768]
        # hiddens[:,0,:] 原始CLS
        # raw_outputs.pooler_output CLS+FNN+Tanh
        cls_feats = hiddens[:, 0, :]  # 获取句子表示CLS

        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))  # 如果是个二分类输出为[16,2]
        else:
            label_feats = hiddens[:, 1:self.num_classes + 1, :]  # 获取所有的标签tokens
            # 分类器: 将自己的文本与自己的各个标签进行内积,哪个内积值最大,就预测哪个值
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)  # ? 爱因斯坦求和约定 输出[16,2]

        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }

        # predicts [16,6]
        # cls_feats [16,768]
        # label_feats [16,6,768]
        return outputs
