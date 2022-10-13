import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss, NewLoss1a, NewLoss1b
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')
        # 使用了什么预训练模型和什么损失函数方法来解决几分类的问题
        # 创建模型
        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    # 打印超参数信息
    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    # 开始训练
    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0

        # 开启训练模式
        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            # 调整文本输入、文本标签、输出值、损失函数
            # inputs 得到是dataloader中text_ids的返回值,即:input_ids,token_type_ids,attention_mask
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            # 使用该训练模型
            outputs = self.model(inputs)
            # outputs 为字典类型,包括predicts,clas_feats,label_feats
            # targets 为torch.Size([16])预测类别的最终数据
            loss = criterion(outputs, targets)

            # 正常的反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练、预测集的损失率、准确率
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    # 评估模型情况,输入为测试集和使用损失函数的方法
    # 和训练函数中的主要区别是没有了BP过程
    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        # 开启评估模式,等价于self.train(False)
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    # 神经网络参数输入、mini-batch训练方式
    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'nl1a':
            criterion = NewLoss1a(self.args.alpha, self.args.temp)
        elif self.args.method == 'nl1b':
            criterion = NewLoss1b(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        best_loss, best_acc = 0, 0

        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))


if __name__ == '__main__':
    logging.set_verbosity_error()

    # 预设参数获取
    args, logger = get_config()

    # 将参数输入到模型中
    ins = Instructor(args, logger)

    # 模型训练评估
    ins.run()
