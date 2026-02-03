from sklearn import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking, tf_encoder, tf_decoder, FT_classifier, FrequencyAdaptationLayer, Signal_Recover, masking2, segment_mask_v1, segment_mask_v3
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl, Temporal_EntropyLoss_v1, Bank_info_max_anchor_min, CORAL, SupConLoss
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from algorithms import clustering
from algorithms.cdd import CDD
from algorithms.sinkhorn import ClassWiseSinkhornLoss
from sklearn.mixture import GaussianMixture
from torchmetrics import Accuracy
import copy

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class SHOT(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(SHOT, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # Freeze the classifier
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # obtain pseudo labels for each epoch
            pseudo_labels = self.obtain_pseudo_labels(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                # prevent gradient accumulation
                self.optimizer.zero_grad()

                # Extract features
                trg_feat, _ = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)

                # pseudo labeling loss
                pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
                target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

                # Entropy loss
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))

                #  Information maximization loss
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

                # Total loss
                loss = entropy_loss + self.hparams['target_cls_wt'] * target_loss

                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': target_loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)

                features, _ = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)

        preds = torch.cat((preds))
        feas = torch.cat((feas))

        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)

        self.feature_extractor.train()
        self.classifier.train()
        return pred_label

class AaD(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(AaD, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            # defining best and last model
            best_src_risk = float('inf')
            best_model = self.network.state_dict()
            last_model = self.network.state_dict()

            fea_bank, score_bank = self.build_feat_score_bank(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                num_samples = len(trg_dataloader.dataset)

                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)

                # output softmax probs
                softmax_out = nn.Softmax(dim=1)(predictions)

                alpha = (1 + 10 * step / self.hparams["num_epochs"] * len(trg_dataloader)) ** (-self.hparams['beta']) * \
                        self.hparams['alpha']
                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                # start gradients
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

                mask = torch.ones((trg_x.shape[0], trg_x.shape[0]))
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = softmax_out.T  # .detach().clone()#

                dot_neg = softmax_out @ copy  # batch x batch

                dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
                neg_pred = torch.mean(dot_neg)
                loss += neg_pred * alpha

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        return last_model, best_model

    def build_feat_score_bank(self, data_loader):
        fea_bank = torch.empty(0).cuda()
        score_bank = torch.empty(0).cuda()

        self.feature_extractor.eval()
        self.classifier.eval()

        # Process data batch by batch
        for data in data_loader:
            batch_data = data[0].cuda()  # Assuming the first element in your batch is the data. Adjust as needed.

            batch_feat,_ = self.feature_extractor(batch_data)
            norm_feats = F.normalize(batch_feat)
            batch_pred = self.classifier(batch_feat)
            batch_probs = nn.Softmax(dim=-1)(batch_pred)

            # Update the banks
            fea_bank = torch.cat((fea_bank, norm_feats.detach()), 0)
            score_bank = torch.cat((score_bank, batch_probs.detach()), 0)

        return fea_bank, score_bank

class NRC(Algorithm):
    """
    Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NIPS 2021)
    https://github.com/Albert0147/NRC_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(NRC, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).cuda()
                softmax_out = nn.Softmax(dim=1)(predictions)

                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                    fea_near = fea_bank[idx_near]  # batch x K x num_dim
                    fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                    distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                    _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                  k=5 + 1)  # M near neighbors for each of above K ones
                    idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                    trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
                    match = (
                            idx_near_near == trg_idx_).sum(-1).float()  # batch x K
                    weight = torch.where(
                        match > 0., match,
                        torch.ones_like(match).fill_(0.1))  # batch x K

                    weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                            5)  # batch x K x M
                    weight_kk = weight_kk.fill_(0.1)

                    # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                    # weight_kk[idx_near_near == trg_idx_]=0

                    score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                    # print(weight_kk.shape)
                    weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                            -1)  # batch x KM

                    score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                    self.configs.num_classes)  # batch x KM x C

                    score_self = score_bank[trg_idx]

                # start gradients
                output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                            -1)  # batch x C x 1
                const = torch.mean(
                    (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                     weight_kk.cuda()).sum(
                        1))  # kl_div here equals to dot product since we do not use log for score_near_kk
                loss = torch.mean(const)

                # nn
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss += torch.mean(
                    (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

                # self, if not explicitly removing the self feature in expanded neighbor then no need for this
                # loss += -torch.mean((softmax_out * score_self).sum(-1))

                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax *
                                          torch.log(msoftmax + self.hparams['epsilon']))
                loss += gentropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=self.configs.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=self.configs.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['entropy_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['entropy_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class TemSR(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TemSR, self).__init__(configs)


        # backbone.
        self.feature_extractor = backbone(configs)
        # classifier.
        self.classifier = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # recover module used in target domain.
        self.signal_recover = Signal_Recover(configs, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )


        self.recover_optimizer = torch.optim.Adam(
            self.signal_recover.parameters(),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1)
        self.mmd_loss = CORAL()  # as hparameter .


    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)   # [batch, num_nodes, feat_dim]

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                src_cls_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # copy one fixed source domain feature extractor.
        src_feature_extractor = copy.deepcopy(self.feature_extractor)

        # freeze both classifier and source feature extractor
        for k, v in src_feature_extractor.named_parameters():
            v.requires_grad = False
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        src_like_epochs = self.hparams["src_like_epochs"]  # Number of epochs for src_like_entropy
        trg_disc_epochs = self.hparams["trg_disc_epochs"]  # Number of epochs for trg_disc_loss

        # sample_bank, entropy_bank = self.build_entropy_bank(trg_dataloader, self.signal_recover,
        #                                                     src_feature_extractor, self.classifier)
        num_sample = len(trg_dataloader.dataset)
        sample_bank = torch.zeros(num_sample, self.configs.input_channels, self.configs.sequence_len).to(self.device)
        entropy_bank = torch.ones(num_sample).to(self.device)

        # print(len(trg_dataloader))

        src_feature_extractor.eval()
        self.classifier.eval()
        self.signal_recover.train()
        self.feature_extractor.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):
                print(step)

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.recover_optimizer.zero_grad()

                # masking and recover.
                masked_data, [mask, mask_indices] = masking2(trg_x, num_splits=self.hparams['num_splits'],
                                                             num_masked=min(self.hparams['num_masked'],self.hparams['num_splits']))

                recovered_data = self.signal_recover(masked_data)

                # recovered and original samples maximization

                recovered_feat, _ = src_feature_extractor(recovered_data) # recovered features.
                trg_max_min_loss, recovered_data_entropy = Bank_info_max_anchor_min(src_feature_extractor, trg_x, recovered_data, recovered_feat,
                                                                                                              entropy_bank, sample_bank,
                                                                                                              self.classifier, self.hparams['anchor_percent'],
                                                                                                              self.hparams['CL_temp'], self.hparams['detach'])
                ### Update the bank
                entropy_bank[trg_idx] = recovered_data_entropy.detach().clone()
                sample_bank[trg_idx] = recovered_data.detach().clone()

                coeffi_anchor = self.hparams["trg_max_min_loss_wt"]*((epoch-1) / self.hparams["num_epochs"])

                raw_feat, raw_feat_seq = self.feature_extractor(trg_x)

                trg_disc_loss = self.mmd_loss(recovered_feat, raw_feat)  # discrepancy loss. (min)


                trg_pred = self.classifier(raw_feat)
                # Entropy loss for target
                trg_ent_loss = EntropyLoss(trg_pred) # entropy loss (min)
                # Computing Gent
                if self.hparams['Gent']:
                    softmax_out = nn.Softmax(dim=1)(trg_pred)
                    msoftmax = softmax_out.mean(dim=0)
                    trg_ent_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                ## Entropy loss for source-like samples

                src_like_pred = self.classifier(recovered_feat)

                segmented_samples_ls = []
                segmented_sample_mask = segment_mask_v1(recovered_data, mask_indices,
                                                        num_segments=self.hparams['num_splits'],
                                                        num_removed=min(self.hparams['num_masked'],self.hparams['num_splits']))

                segmented_sample_first = segment_mask_v3(recovered_data, posi = 'first', padding=True,
                                                         num_segments=self.hparams['num_splits'],
                                                         num_removed=min(self.hparams['num_masked'],
                                                                         self.hparams['num_splits']))

                segmented_sample_last = segment_mask_v3(recovered_data, posi='last', padding=True,
                                                        num_segments=self.hparams['num_splits'],
                                                        num_removed=min(self.hparams['num_masked'],
                                                                        self.hparams['num_splits']))
                segmented_samples_ls.append(segmented_sample_mask)
                segmented_samples_ls.append(segmented_sample_first)
                segmented_samples_ls.append(segmented_sample_last)
                segmented_samples_ls = torch.cat(segmented_samples_ls, 0)
                segmented_feats, _  = src_feature_extractor(segmented_samples_ls)
                segmented_pred = self.classifier(segmented_feats)
                segmented_all_pred = torch.cat((src_like_pred, segmented_pred),0)
                src_like_entropy = Temporal_EntropyLoss_v1(segmented_all_pred, src_like_pred.size(0))

                if self.hparams['Gent']:
                    softmax_out_src_like = nn.Softmax(dim=1)(src_like_pred)
                    msoftmax_src_like = softmax_out_src_like.mean(dim=0)
                    src_like_entropy -= torch.sum(-msoftmax_src_like * torch.log(msoftmax_src_like + 1e-5))
                total_cycle_epochs = src_like_epochs + trg_disc_epochs
                current_cycle_epoch = (epoch - 1) % total_cycle_epochs

                if current_cycle_epoch < src_like_epochs:
                    # Train with src_like_entropy for the specified epochs
                    loss = self.hparams['src_like_entropy_wt'] * src_like_entropy
                    loss += coeffi_anchor * trg_max_min_loss

                else:
                    # Train with trg_disc_loss for the next set of epochs
                    loss = self.hparams['disc_loss_wt'] * trg_disc_loss

                # Overall objective loss (include trg_ent_loss in every epoch)
                loss += self.hparams['ent_loss_wt'] * trg_ent_loss

                loss.backward()
                self.optimizer.step()
                self.recover_optimizer.step()

                losses = {'entropy_loss': trg_ent_loss.detach().item(), 'discrepancy_loss': trg_disc_loss.detach().item(),
                          'src_like_entropy_loss': src_like_entropy.detach().item(),'total_loss':loss.detach().item(),
                          'trg_max_min_loss': trg_max_min_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, src_like_pred.size(0))

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
            # last_model = deepcopy(self.network.state_dict(=))
        return last_model, best_model

class CESFDA(Algorithm):

    def __init__(self, backbone, configs, hparams, device, is_wegited=False):
        super(CESFDA, self).__init__(configs)
        self.is_wegited = is_wegited
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier_aad = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_aad = torch.optim.Adam(
            self.classifier_aad.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.supcon = SupConLoss(self.device)
    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()
        classifier_state_dict = self.classifier.state_dict()
        self.classifier_aad.load_state_dict(classifier_state_dict)
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        num_samples = len(trg_dataloader.dataset)
        fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len) 
        score_bank = torch.randn(num_samples, self.configs.num_classes).cuda()                              
        temp = True   
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (trg_x, y, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                if temp:
                    self.feature_extractor.eval()
                    self.classifier_aad.eval()
                    with torch.no_grad():
                        iter_test = iter(trg_dataloader)
                        for i in range(len(trg_dataloader)):
                            data = next(iter_test)
                            inputs = data[0]
                            indx = data[-1]
                            inputs = inputs.cuda()
                            output, _ = self.feature_extractor(inputs)
                            output_norm = F.normalize(output)
                            outputs = self.classifier_aad(output)
                            outputs = nn.Softmax(-1)(outputs)
                            fea_bank[indx] = output_norm.detach().clone().cpu()
                            score_bank[indx] = outputs.detach().clone()  # .cpu()
                    self.feature_extractor.train()
                    self.classifier_aad.train()
                    temp = False
                self.optimizer.zero_grad()
                self.optimizer_aad.zero_grad()
                trg_feat, _ = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)#shot logits
                trg_pred_aad = self.classifier_aad(trg_feat)#AaD logits
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                softmax_out_aad = nn.Softmax(dim=1)(trg_pred_aad)
                with torch.no_grad():
                    output_f_norm = F.normalize(trg_feat)
                    output_f_ = output_f_norm.cpu().detach().clone()
                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out_aad.detach().clone()
                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C
                softmax_out_un = softmax_out_aad.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C
                loss_aad = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))
                with torch.no_grad():
                    entropy1 = -torch.sum(softmax_out * torch.log2(softmax_out + 1e-5), dim=1)
                    entropy2 = -torch.sum(softmax_out_aad * torch.log2(softmax_out_aad + 1e-5), dim=1)
                    weight = torch.exp(-entropy1) / (torch.exp(-entropy1) + torch.exp(-entropy2))
                    weighted_logits = weight[:, None] * trg_pred + (1 - weight)[:, None] * trg_pred_aad
                    weighted_softmax = nn.Softmax(dim=1)(weighted_logits)
                    max_entropy = torch.log2(torch.tensor(self.configs.num_classes))
                    # w = -torch.sum(weighted_softmax * torch.log2(weighted_softmax + 1e-5), dim=1)
                    # w = w / max_entropy
                    # w = torch.exp(-w)

                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))
                if self.is_wegited:
                    _, pseudo_label = weighted_softmax.max(dim=1)
                else:
                    with torch.no_grad():
                        max_logits = []
                        for i in range(softmax_out.shape[0]):
                            v1 = softmax_out[i].max()
                            v2 = softmax_out_aad[i].max()
                            if v1 > v2:
                                max_logits.append(softmax_out[i])
                            else:
                                max_logits.append(softmax_out_aad[i])
                        max_logits = torch.cat([i.unsqueeze(0) for i in max_logits], dim=0)
                        pseudo_label = max_logits.argmax(-1)
                target_loss = self.cross_entropy(trg_pred.squeeze(), pseudo_label.long())
                loss = entropy_loss + self.hparams['aad_wt'] * loss_aad + self.hparams['target_cls_wt'] * target_loss
                
                loss.backward()
                self.optimizer.step()
                self.optimizer_aad.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        return last_model, best_model

class MAPU_Anchor_freq_gmm(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU_Anchor_freq_gmm, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        amp_scale = hparams.get('amp_scale', 0.2)
        phase_scale = hparams.get('phase_scale', 0.2)
        self.freq_adapter = FrequencyAdaptationLayer(
            seq_len=configs.sequence_len,
            channels=configs.input_channels,
            amp_scale=amp_scale,
            phase_scale=phase_scale,
            hidden_dim=64
        ).to(device)
        self.network = nn.Sequential(self.freq_adapter, self.feature_extractor, self.classifier)
        # self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )


    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                # src_x = self.freq_adapter(src_x)
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=self.configs.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)
                                 
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):
        best_src_risk = float('inf')
        best_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.feature_extractor.named_parameters():
            v.requires_grad = False
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_x = self.freq_adapter(trg_x)
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=self.configs.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob)) 


                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))
                
                '''
                Overall objective loss
                '''
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss
                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['entropy_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['entropy_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        last_model = deepcopy(self.network.state_dict())
        return last_model, best_model
