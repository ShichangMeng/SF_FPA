import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from math import ceil
import numpy as np

class DIST(object):
    """
    DIST 类用于计算特征向量之间的距离。支持余弦距离。

    参数:
        dist_type (str): 距离类型，目前支持 'cos'（余弦距离）。
    """

    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        """
        计算两个特征点或特征集之间的距离。

        参数:
            pointA (Tensor): 特征点或特征集A。
            pointB (Tensor): 特征点或特征集B。
            cross (bool): 是否计算交叉距离（即A和B之间的成对距离）。

        返回:
            Tensor: 计算得到的距离值。
        """
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        """
        计算余弦距离。

        参数:
            pointA (Tensor): 特征点或特征集A。
            pointB (Tensor): 特征点或特征集B。
            cross (bool): 是否计算交叉距离。

        返回:
            Tensor: 余弦距离值。
        """
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    """
    Clustering 类用于在特征空间中对样本进行聚类。

    参数:
        eps (float): 聚类停止的阈值，当中心的变化小于此值时停止聚类。
        feat_key (str): 特征在模型输出中的键名，用于提取特征。
        max_len (int): 每次处理的最大样本数量。
        dist_type (str): 使用的距离类型。
    """

    def __init__(self, eps, feat_key, max_len=1000, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.data2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        """
        设置初始的聚类中心。

        参数:
            init_centers (Tensor): 初始中心。
        """
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        """
        判断是否停止聚类。

        参数:
            centers (Tensor): 当前的聚类中心。

        更新:
            self.stop (bool): 根据中心变化决定是否停止聚类。
        """
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        """
        为每个样本分配聚类标签。

        参数:
            feats (Tensor): 样本特征。

        返回:
            Tensor, Tensor: 样本到聚类中心的距离和对应的标签。
        """
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def align_centers(self):
        """
        使用线性分配算法对齐当前聚类中心和初始中心。

        返回:
            ndarray: 对齐后的中心索引。
        """
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, weights, loader):
        """
        收集样本的特征、标签和路径信息。

        参数:
            net (nn.Module): 深度学习模型。
            loader (DataLoader): 数据加载器。
        """
        data_feat, data_gt, data_all = [], [], []
        for sample in iter(loader):
            data = sample[0].cuda()
            data_all += [data]
            data_gt += [(sample[3].cuda())]
            feature, _ = net.forward(data)[0].data
            data_feat += [feature]
        self.samples['data'] = torch.cat(data_all, dim=0).float()
        self.samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        self.samples['feature'] = torch.cat(data_feat, dim=0)

    def feature_clustering(self, samples):

        centers = None
        self.stop = False
        self.samples = samples
        feature = self.samples['feature']

        refs = (torch.LongTensor(range(self.num_classes)).unsqueeze(1)).cuda()
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop:
                break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                identity = (torch.eye(self.num_classes)).cuda()
                labels_onehot = torch.index_select(identity, 0, labels)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center, labels = [], []
        start = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)
            identity = (torch.eye(self.num_classes)).cuda()
            labels_onehot = torch.index_select(identity, 0, cur_labels)
            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['clu_label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        self.centers = self.centers[cluster2label, :]
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['clu_label'][k] = cluster2label[self.samples['clu_label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, self.init_centers))

        for i in range(num_samples):
            self.data2label[self.samples['data'][i]] = self.samples['clu_label'][i].item()

        # del self.samples['feature']

