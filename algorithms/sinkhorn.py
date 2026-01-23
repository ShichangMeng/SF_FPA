import torch
import torch.nn as nn

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            'sum': the output will be summed. Default: 'none'
            
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        
        # Handle batch size
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).to(x.device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).to(x.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            
            # log-domain updates for numerical stability
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            
            # Check convergence
            err = (u - u1).abs().sum(-1).mean()
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C


class ClassWiseSinkhornLoss(nn.Module):

    def __init__(self, eps=0.1, max_iter=100, num_classes=6):
        super(ClassWiseSinkhornLoss, self).__init__()
        self.num_classes = num_classes
        # 初始化基础 Sinkhorn 模块
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter, reduction='mean')

    def forward(self, src_feat_sampled, trg_feat, plabels):
        """
        Args:
            src_feat_sampled: (Num_Classes, Samples_Per_Class, Feat_Dim) 
                              或者 (Total_Source_Samples, Feat_Dim)
            trg_feat: (Batch_Size, Feat_Dim) 目标域特征
            plabels: (Batch_Size) 目标域的伪标签
        """
        loss = 0.0
        valid_classes = 0
        
        # 确保 plabels 是 Tensor 且设备正确
        if not torch.is_tensor(plabels):
            plabels = torch.tensor(plabels).to(trg_feat.device)
        else:
            plabels = plabels.to(trg_feat.device)

        # 遍历每个类别进行对齐
        for c in range(self.num_classes):
            # 1. 获取目标域当前类别的特征
            trg_idx = (plabels == c).nonzero(as_tuple=True)[0]
            
            # 【核心判空】如果当前 Batch 里没有这个类别的样本，跳过
            if len(trg_idx) == 0:
                continue
            
            trg_feat_c = trg_feat[trg_idx]

            # 2. 获取源域当前类别的特征
            # 假设 src_feat_sampled 是形状为 [Num_Classes, N_Samples, Dim] 的 Tensor
            # 如果你的源域特征是扁平的，你需要在这里传对应的 index
            src_feat_c = src_feat_sampled[c] 
            
            # 【再次判空】确保源域采样也没有空（虽然通常源域是满的）
            if src_feat_c.size(0) == 0:
                continue

            # 3. 计算该类别的 Sinkhorn 距离
            # Sinkhorn 要求输入至少是 2D (N, D)
            # 这里的 cost 是标量 (reduction='mean')
            cost, _, _ = self.sinkhorn(src_feat_c, trg_feat_c)
            
            loss += cost
            valid_classes += 1
        
        # 4. 平均 Loss
        if valid_classes > 0:
            loss = loss / valid_classes
        else:
            # 如果所有类别都空了（极罕见），返回0梯度张量
            loss = torch.tensor(0.0, device=trg_feat.device, requires_grad=True)
            
        return loss