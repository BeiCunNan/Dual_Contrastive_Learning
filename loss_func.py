import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 交叉熵损失函数
class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.xent_loss(outputs['predicts'], targets)

# 输入情况
# anchor锚点 [16,768]
# target [16,768]
# labels [16]

# 如果使用的是SCL方法,anchor和target都是文本表示CLS
# 如果使用的是DualCL,anchor和target为文本CLS和正样本的标签表示

class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # α
        self.temp = temp  # 温度系数

    def nt_xent_loss(self, anchor, target, labels):
        # print('1',anchor)
        # print('2',anchor.shape)
        # print('3',target)
        # print('4',target.shape)
        # print('5',labels)
        # print('6',labels.shape)

        # 作用：作用找到与自己标签相同的对象
        # 返回：mask[16,16]的矩阵
        # 用法：如想看第2条数据的同类,只要找同一行中为True的位置(主对角线元素已设置为False)
        with torch.no_grad():
            # 输入的lables -- torch.Size([16])
            labels = labels.unsqueeze(-1)  # 升维,一维变二维 -- torch.Size([16, 1])
            # labels.transpose(0, 1) 为转置, -- torch.Size([1,16])
            mask = torch.eq(labels, labels.transpose(0, 1))  # 对两个元素进行逐一检测,如果元素相同则True,不同为False -- torch.Size([16, 16])
            # delete diag elem
            # torch.diag(mask)是一个[16]全是True的向量,即将主对角线的元素拿出来形成一个向量
            # torch.diag_embed(torch.diag(mask))将向量放到主对角线上,最终表现形式为一个[16,16]的单位矩阵
            # ^ 是对称差的意思,因此下面一行代码的作用是将mask对角线上的数据设置为False
            mask = mask ^ torch.diag_embed(torch.diag(mask))  # torch.Size([16, 16])

        # 作用:anchor与每个target求内积，删除对角线元素，
        # anchor和target求内积,并删除对角线元素,全部元素减去本行最大的元素值-->全部变成非正数,最后取指数
        # anchor [16,768] target [16,768] anchor_dot_target [16,16] [i,j]表示anchor的第i行和target的第j行进行内积
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))  # 删除对角线元素

        # 作用: 将内积结果进一步处理，每一行数据减去每一行的最大值,全部变成非正数 ???
        # 返回: logits[16,16]
        # dim: 输出每一行的最大值 keepdim:返回值形式为[n,1]即[16,1],即获得与该anchor锚点最相关的target
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)  # logits [16,16] 所有的数据减去最大值
        logits = anchor_dot_target - logits_max.detach()  # detach为切断该分支的反向传播,说明该部分的网络参数我们想保存不想进行修改

        # 作用: 取指数
        # 公式: 分母指数部分
        exp_logits = torch.exp(logits)  # [16,16]

        # 作用: 正样本全部找出来,负样本全部变成0 [16,16]
        logits = logits * mask

        # [16,16]
        # 作用: log 函数部分
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 作用: 计算每个样本与自己同类的个数,如果没有同类就设置为1
        # 公式: 对应|Pi|
        # 返回: mask_sum [16]
        # in case that mask.sum(1) is zero   [16]
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        # compute log-likelihood [16]

        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()

        # 作用: 一个Epoch中所有样本的损失函数取平均
        # 公式: 对应的是 1/N sum
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss


class DualLoss(SupConLoss):

    def __init__(self, alpha, temp):

        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        # [16, 6, 768]
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)

        # 作用: 获取本样本的积极样本
        # torch.gather为选择功能,按照dim=1,行来选择
        # reshape(-1,1,1)将二维数组转化为一维一列的数组
        # [16, 768]
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)
        return ce_loss + cl_loss_1 + cl_loss_2


class PosLoss(SupConLoss):

    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        # [16, 6, 768]
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)

        # 作用: 获取本样本的积极样本
        # torch.gather为选择功能,按照dim=1,行来选择
        # reshape(-1,1,1)将二维数组转化为一维一列的数组
        # [16, 768]
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        pos_loss_1 = self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_pos_label_feats, targets)
        return ce_loss + pos_loss_1


class NewLossmm(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)
        return ce_loss + cl_loss_1 + cl_loss_2


class NewLoss1a(SupConLoss):

    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        # [16, 6, 768]
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)

        # 作用: 获取本样本的积极样本
        # torch.gather为选择功能,按照dim=1,行来选择
        # reshape(-1,1,1)将二维数组转化为一维一列的数组
        # [16, 768]
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_pos_label_feats, targets)
        cl_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss_1 + cl_loss_2


class NewLoss1b(nn.Module):
    def __init__(self, alpha1b, temp1b):
        super().__init__()
        self.xent_loss1b = nn.CrossEntropyLoss()
        self.alpha1b = alpha1b
        self.temp1b = temp1b

    def nt_xent_loss1b(self, anchor1, anchor2, target1, target2, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))

        # compute logits
        anchor1_dot_target1 = torch.einsum('bd,cd->bc', anchor1, target1) / self.temp1b
        anchor2_dot_target2 = torch.einsum('bd,cd->bc', anchor2, target2) / self.temp1b
        anchor_dot_target = (anchor1_dot_target1 + anchor2_dot_target2) / 2

        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        ce_loss = (1 - self.alpha1b) * self.xent_loss1b(outputs['predicts'], targets)

        nl_loss = self.alpha1b * self.nt_xent_loss1b(normed_cls_feats, normed_pos_label_feats, normed_cls_feats,
                                                 normed_pos_label_feats, targets)

        return ce_loss + nl_loss

class NewLoss2a(SupConLoss):

    def __init__(self, alpha, temp):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        # [16, 6, 768]
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)

        # 作用: 获取本样本的积极样本
        # torch.gather为选择功能,按照dim=1,行来选择
        # reshape(-1,1,1)将二维数组转化为一维一列的数组
        # [16, 768]
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        dual_loss_1 = 0.25 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        dual_loss_2 = 0.25 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, targets)
        cl_loss_1 = 0.25 * self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_pos_label_feats, targets)
        cl_loss_2 = 0.25 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + dual_loss_1 + dual_loss_2 + cl_loss_1 + cl_loss_2


class NewLoss2b(nn.Module):
    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor1, anchor2, anchor3, target1, target2, target3, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))

        # compute logits
        anchor1_dot_target1 = torch.einsum('bd,cd->bc', anchor1, target1) / self.temp
        anchor2_dot_target2 = torch.einsum('bd,cd->bc', anchor2, target2) / self.temp
        anchor3_dot_target3 = torch.einsum('bd,cd->bc', anchor3, target3) / self.temp
        anchor_dot_target = (anchor1_dot_target1 + anchor2_dot_target2 + anchor3_dot_target3) / 3

        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        nl2b_loss_1 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats, normed_cls_feats,
                                                           normed_cls_feats,
                                                           normed_pos_label_feats, normed_pos_label_feats, targets)
        nl2b_loss_2 = 0.5 * self.alpha * self.nt_xent_loss(normed_cls_feats, normed_pos_label_feats,
                                                           normed_pos_label_feats, normed_cls_feats,
                                                           normed_pos_label_feats, normed_cls_feats, targets)

        return ce_loss + nl2b_loss_1 + nl2b_loss_2


class NewLoss3a(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))

        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        adt_length=anchor_dot_target.size(0)
        tt = torch.tensor(np.arange(anchor_dot_target.size(0))).view(adt_length).long().cuda()

        loss_1 = self.xent_loss(anchor_dot_target, tt)

        loss_2 = self.xent_loss(anchor_dot_target.transpose(0, 1), tt)

        return loss_1 + loss_2

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss_1 = self.alpha * self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss_1

class NewLoss3b(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))

        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        adt_length=anchor_dot_target.size(0)
        tt = torch.tensor(np.arange(anchor_dot_target.size(0))).view(adt_length).long().cuda()

        loss_1 = self.xent_loss(anchor_dot_target, tt)

        loss_2 = self.xent_loss(anchor_dot_target.transpose(0, 1), tt)

        return loss_1 + loss_2

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1,
                                                                                                                normed_label_feats.size(
                                                                                                                    -1))).squeeze(
            1)

        cl_loss_1 = self.nt_xent_loss(normed_pos_label_feats, normed_cls_feats, targets)
        return cl_loss_1


