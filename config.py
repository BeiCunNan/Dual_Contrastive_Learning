import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime


# 1

def get_config():
    parser = argparse.ArgumentParser()  # 参数解析器,类似一个容器
    num_classes = {'sst2': 2, 'subj': 2, 'trec': 6, 'pc': 2, 'cr': 2, 'sst5': 5, 'yelp5': 5, "dp": 14, "ag": 4}
    ''' Base '''
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='cr', choices=num_classes.keys())
    parser.add_argument('--model_name', type=str, default='bert', choices=['bert', 'roberta','wsp'])
    parser.add_argument('--method', type=str, default='nl2b',
                        choices=['ce', 'scl', 'dualcl', 'nl', 'nl1a', 'nl1b', 'nl2a', 'nl2b', 'nl3a', 'nl3b',
                                 'pos'])  # CE为交叉熵、SCL为标准对比学习
    ''' Optimization '''
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=0.01)  # 衰变率
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temp', type=float, default=0.1)

    ''' Environment '''
    parser.add_argument('--backend', default=False,
                        action='store_true')  # 触发工具,若使用了backend,则python config.py返回值为true,反之为false
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    parser.add_argument('--device', type=str,
                        default='cuda')  # parser.add_argument('--device', type=str, default='cuda' )

    args = parser.parse_args()  # 创建parser实例,将所有参数返回该实例中
    args.num_classes = num_classes[args.dataset]  # 因为是dict类型,不太方便放到add_argument中，因此需要将其单独放出来
    args.device = torch.device(args.device)

    '''log'''
    args.log_name = '{}_{}_{}_{}.log'.format(args.dataset, args.model_name, args.method,
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
