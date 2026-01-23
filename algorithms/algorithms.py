import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking, tf_encoder, tf_decoder, FT_classifier, FrequencyAdaptationLayer
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from algorithms import clustering
from algorithms.cdd import CDD
from algorithms.sinkhorn import ClassWiseSinkhornLoss
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

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
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


class TF_MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TF_MAPU, self).__init__(configs)

        self.feature_extractor = tf_encoder(configs).to(device)
        self.decoder = tf_decoder(configs).to(device)
        self.classifier = FT_classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
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
        self.recons = nn.L1Loss(reduction='sum').to(device)

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat, out_s = self.feature_extractor(src_x)
                src_recon = self.decoder(src_feat, out_s)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask, out_s_mask = self.feature_extractor(masked_data)
                recons = 1e-4 * self.recons(src_recon, src_x)
                                 
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + recons
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        weight = self.classifier.logits.weight.data 
        self.source_anchors = F.normalize(weight, dim=1).cpu()
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()

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
                trg_feat, trg_feat_seq, out_trg = self.feature_extractor(trg_x)
                trg_recon = self.decoder(trg_feat, out_trg)
                recons = 1e-4 * self.recons(trg_recon, trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask, out_trg_mask = self.feature_extractor(masked_data)

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
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss + recons

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item()}
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

class TF_MAPU_Anchor(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TF_MAPU_Anchor, self).__init__(configs)

        self.feature_extractor = tf_encoder(configs).to(device)
        self.decoder = tf_decoder(configs).to(device)
        self.classifier = FT_classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
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
        self.recons = nn.L1Loss(reduction='sum').to(device)

        self.clustering = clustering.Clustering(
            0.001,
            'feat',
            10
        )
        self.cdd = CDD(kernel_num=hparams.get('kernel_num', [5, 5]), 
                       kernel_mul=hparams.get('kernel_mul', [2.0, 2.0]),
                       num_layers=1, 
                       num_classes=self.configs.num_classes, 
                       intra_only=False)
        self.source_anchors = None
        self.surrogate_feature_sampler = {}

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat, out_s = self.feature_extractor(src_x)
                src_recon = self.decoder(src_feat, out_s)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask, out_s_mask = self.feature_extractor(masked_data)
                recons = 1e-4 * self.recons(src_recon, src_x)
                                 
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + recons
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        weight = self.classifier.logits.weight.data 
        self.source_anchors = F.normalize(weight, dim=1).cpu()
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):
        samples = self.collect_samples(trg_dataloader)
        init_target_centers = self.classifier.logits.weight.data.detach().clone()
        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(samples)
        logger.info("Constructing surrogate feature samplers...")
        filtered_classes = list(range(self.configs.num_classes))
        self.construct_surrogate_feature_sampler(samples, filtered_classes)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                with torch.no_grad():
                    feat_temp, _, _ = self.feature_extractor(trg_x)
                    output_temp = self.classifier(feat_temp)
                    pseudo_labels = output_temp.argmax(dim=1)
                    # 统计当前 batch 各个类别的数量
                    nums_cls = [torch.sum(pseudo_labels == c).item() for c in range(self.configs.num_classes)]

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq, out_trg = self.feature_extractor(trg_x)
                trg_recon = self.decoder(trg_feat, out_trg)
                recons = 1e-4 * self.recons(trg_recon, trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask, out_trg_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)
                
                src_feat_sampled = self.sample_source_features(nums_cls, self.configs.num_classes)
                src_feat_sampled = src_feat_sampled.to(self.device)
                cdd_loss = self.cdd.forward([src_feat_sampled], [trg_feat], nums_cls, nums_cls)['cdd']
                cdd_loss *= self.hparams.get('cdd_weight', 1.0)

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
                loss = cdd_loss + trg_ent + self.hparams['TOV_wt'] * tov_loss + recons

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item(), 'cdd_loss': cdd_loss.detach().item()}
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
    
    def collect_samples(self, loader):
        """
        收集样本的特征、标签和路径信息。

        参数:
            net (nn.Module): 深度学习模型。
            loader (DataLoader): 数据加载器。
        """
        samples = {}
        data_feat, data_gt, data_all, labels = [], [], [], []
        for x, gt, _ in iter(loader):
            data = x.float().to(self.device)
            data_all += [data]
            data_gt += [gt.long().to(self.device)]
            feature = self.feature_extractor.forward(data)[0]
            data_feat += [feature]
            label = self.classifier.forward(feature).argmax(dim=1)
            labels += [label]
        samples['data'] = torch.cat(data_all, dim=0).float()
        samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        samples['feature'] = torch.cat(data_feat, dim=0)
        samples['label'] = torch.cat(labels, dim=0)
        return samples
    
    def construct_surrogate_feature_sampler(self, samples, filtered_classes):
        """构建高斯采样器"""
        feats = samples['feature']
        # 这里的 label 来自聚类的伪标签 results
        plabels = self.clustering.samples['label'] 
        
        variance_mult = self.hparams.get('variance_mult', 1.0)
        
        # 计算 Target 的均值和方差
        target_mean = torch.zeros(self.configs.num_classes, feats.shape[-1])
        target_std = torch.zeros(self.configs.num_classes, feats.shape[-1])

        for c in filtered_classes:
            idx = torch.where(plabels == c)[0]
            if len(idx) > 1:
                _std, _mean = torch.std_mean(feats[idx], dim=0, unbiased=True)
                target_mean[c] = _mean
                target_std[c] = _std

        # 构建采样器
        for i in range(self.configs.num_classes):
            cur_target_norm = target_mean[i].norm(p=2)
            if cur_target_norm < 1e-3:
                self.surrogate_feature_sampler[i] = None
                continue
            
            # 【核心逻辑】
            # Mean: 方向取自源域 Anchor，长度取自目标域 (假设分布发生偏移但相对结构保持)
            estimated_source_mean = self.source_anchors[i] * cur_target_norm
            # Std: 使用目标域的 Std
            estimated_source_std = target_std[i]
            estimated_source_std[estimated_source_std == 0] = 1e-4
            
            self.surrogate_feature_sampler[i] = torch.distributions.normal.Normal(
                estimated_source_mean, estimated_source_std * variance_mult
            )

    def sample_source_features(self, nums_cls, num_classes):
        """根据需要的数量采样源域特征"""
        source_features = []
        feat_dim = self.configs.features_len * self.configs.final_out_channels + self.configs.fourier_modes*2/self.configs.stride
        for c in range(num_classes):
            num = nums_cls[c]
            if num > 0 and self.surrogate_feature_sampler.get(c) is not None:
                # 采样
                s = self.surrogate_feature_sampler[c].sample((num,))
                source_features.append(s)
            elif num > 0:
                # 如果该类无法采样（例如目标域该类样本太少），补零或者随机噪音
                source_features.append(torch.zeros(num, int(feat_dim)))
        
        if len(source_features) > 0:
            return torch.cat(source_features, dim=0)
        else:
            # 极少数情况 Batch 为空
            return torch.tensor([]).to(self.device)

class TF_MAPU_Anchor_sink(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TF_MAPU_Anchor_sink, self).__init__(configs)

        self.feature_extractor = tf_encoder(configs).to(device)
        self.decoder = tf_decoder(configs).to(device)
        self.classifier = FT_classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
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
        self.recons = nn.L1Loss(reduction='sum').to(device)

        self.clustering = clustering.Clustering(
            0.001,
            'feat',
            10
        )
        self.sinkhorn_loss_fn = ClassWiseSinkhornLoss(
            eps=0.1, 
            max_iter=100, 
            num_classes=self.configs.num_classes
        )
        self.source_anchors = None
        self.surrogate_feature_sampler = {}

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat, out_s = self.feature_extractor(src_x)
                src_recon = self.decoder(src_feat, out_s)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask, out_s_mask = self.feature_extractor(masked_data)
                recons = 1e-4 * self.recons(src_recon, src_x)
                                 
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + recons
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        weight = self.classifier.logits.weight.data 
        self.source_anchors = F.normalize(weight, dim=1).cpu()
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):
        samples = self.collect_samples(trg_dataloader)
        init_target_centers = self.classifier.logits.weight.data.detach().clone()
        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(samples)
        logger.info("Constructing surrogate feature samplers...")
        filtered_classes = list(range(self.configs.num_classes))
        self.construct_surrogate_feature_sampler(samples, filtered_classes)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                with torch.no_grad():
                    feat_temp, _, _ = self.feature_extractor(trg_x)
                    output_temp = self.classifier(feat_temp)
                    pseudo_labels = output_temp.argmax(dim=1)
                    # 统计当前 batch 各个类别的数量
                    nums_cls = [torch.sum(pseudo_labels == c).item() for c in range(self.configs.num_classes)]

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq, out_trg = self.feature_extractor(trg_x)
                trg_recon = self.decoder(trg_feat, out_trg)
                recons = 1e-4 * self.recons(trg_recon, trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask, out_trg_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)
                
                src_feat_sampled = self.sample_source_features(nums_cls, self.configs.num_classes)
                src_feat_sampled = [sampled.to(self.device) for sampled in src_feat_sampled]
                sinkhorn_loss = self.sinkhorn_loss_fn(src_feat_sampled, trg_feat, pseudo_labels)
                sinkhorn_loss *= self.hparams.get('sink_weight', 1.0)

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
                loss = sinkhorn_loss + trg_ent + self.hparams['TOV_wt'] * tov_loss + recons

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item(), 'sinkhorn_loss': sinkhorn_loss.detach().item()}
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
    
    def collect_samples(self, loader):
        """
        收集样本的特征、标签和路径信息。

        参数:
            net (nn.Module): 深度学习模型。
            loader (DataLoader): 数据加载器。
        """
        samples = {}
        data_feat, data_gt, data_all, labels = [], [], [], []
        for x, gt, _ in iter(loader):
            data = x.float().to(self.device)
            data_all += [data]
            data_gt += [gt.long().to(self.device)]
            feature = self.feature_extractor.forward(data)[0]
            data_feat += [feature]
            label = self.classifier.forward(feature).argmax(dim=1)
            labels += [label]
        samples['data'] = torch.cat(data_all, dim=0).float()
        samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        samples['feature'] = torch.cat(data_feat, dim=0)
        samples['label'] = torch.cat(labels, dim=0)
        return samples
    
    def construct_surrogate_feature_sampler(self, samples, filtered_classes):
        """构建高斯采样器"""
        feats = samples['feature']
        # 这里的 label 来自聚类的伪标签 results
        plabels = self.clustering.samples['label'] 
        
        variance_mult = self.hparams.get('variance_mult', 1.0)
        
        # 计算 Target 的均值和方差
        target_mean = torch.zeros(self.configs.num_classes, feats.shape[-1])
        target_std = torch.zeros(self.configs.num_classes, feats.shape[-1])

        for c in filtered_classes:
            idx = torch.where(plabels == c)[0]
            if len(idx) > 1:
                _std, _mean = torch.std_mean(feats[idx], dim=0, unbiased=True)
                target_mean[c] = _mean
                target_std[c] = _std

        # 构建采样器
        for i in range(self.configs.num_classes):
            cur_target_norm = target_mean[i].norm(p=2)
            if cur_target_norm < 1e-3:
                self.surrogate_feature_sampler[i] = None
                continue
            
            # 【核心逻辑】
            # Mean: 方向取自源域 Anchor，长度取自目标域 (假设分布发生偏移但相对结构保持)
            estimated_source_mean = self.source_anchors[i] * cur_target_norm
            # Std: 使用目标域的 Std
            estimated_source_std = target_std[i]
            estimated_source_std[estimated_source_std == 0] = 1e-4
            
            self.surrogate_feature_sampler[i] = torch.distributions.normal.Normal(
                estimated_source_mean, estimated_source_std * variance_mult
            )

    def sample_source_features(self, nums_cls, num_classes):
        """根据需要的数量采样源域特征"""
        source_features = []
        feat_dim = self.configs.features_len * self.configs.final_out_channels + self.configs.fourier_modes*2/self.configs.stride
        for c in range(num_classes):
            num = nums_cls[c]
            if num > 0 and self.surrogate_feature_sampler.get(c) is not None:
                # 采样
                s = self.surrogate_feature_sampler[c].sample((num,))
                source_features.append(s)
            else:
                # 如果该类无法采样（例如目标域该类样本太少），补零或者随机噪音
                source_features.append(torch.zeros(num, int(feat_dim)))

        return source_features

class TF_MAPU_Anchor_freq(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TF_MAPU_Anchor_freq, self).__init__(configs)

        self.feature_extractor = tf_encoder(configs).to(device)
        self.decoder = tf_decoder(configs).to(device)
        self.classifier = FT_classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.freq_adapter = FrequencyAdaptationLayer(
            seq_len=configs.sequence_len,
            channels=configs.input_channels,
            hidden_dim=64
        ).to(device)
        self.network = nn.Sequential(self.freq_adapter, self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            # list(self.feature_extractor.parameters()) + \
            # list(self.decoder.parameters()) + \
            # list(self.classifier.parameters()) + \
            list(self.freq_adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.decoder.parameters())+\
                list(self.classifier.parameters()),
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
        self.recons = nn.L1Loss(reduction='sum').to(device)

        self.clustering = clustering.Clustering(
            0.001,
            'feat',
            10
        )
        self.cdd = CDD(kernel_num=hparams.get('kernel_num', [5, 5]), 
                       kernel_mul=hparams.get('kernel_mul', [2.0, 2.0]),
                       num_layers=1, 
                       num_classes=self.configs.num_classes, 
                       intra_only=self.configs.intra_only)
        self.source_anchors = None
        self.surrogate_feature_sampler = {}

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat, out_s = self.feature_extractor(src_x)
                src_recon = self.decoder(src_feat, out_s)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=8, num_masked=1)
                src_feat_mask, seq_src_feat_mask, out_s_mask = self.feature_extractor(masked_data)
                recons = 1e-4 * self.recons(src_recon, src_x)
                                 
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + recons
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        weight = self.classifier.logits.weight.data 
        self.source_anchors = F.normalize(weight, dim=1).cpu()
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):
        samples = self.collect_samples(trg_dataloader)
        init_target_centers = self.classifier.logits.weight.data.detach().clone()
        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(samples)
        logger.info("Constructing surrogate feature samplers...")
        filtered_classes = list(range(self.configs.num_classes))
        self.construct_surrogate_feature_sampler(samples, filtered_classes)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                with torch.no_grad():
                    feat_temp, _, _ = self.feature_extractor(trg_x)
                    output_temp = self.classifier(feat_temp)
                    pseudo_labels = output_temp.argmax(dim=1)
                    # 统计当前 batch 各个类别的数量
                    nums_cls = [torch.sum(pseudo_labels == c).item() for c in range(self.configs.num_classes)]

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_x = self.freq_adapter(trg_x)
                trg_feat, trg_feat_seq, out_trg = self.feature_extractor(trg_x)
                trg_recon = self.decoder(trg_feat, out_trg)
                recons = 1e-4 * self.recons(trg_recon, trg_x)

                masked_data, mask = masking(trg_x, num_splits=8, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask, out_trg_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)
                
                src_feat_sampled = self.sample_source_features(nums_cls, self.configs.num_classes)
                src_feat_sampled = src_feat_sampled.to(self.device)
                cdd_loss = self.cdd.forward([src_feat_sampled], [trg_feat], nums_cls, nums_cls)['cdd']
                cdd_loss *= self.hparams.get('cdd_weight', 1.0)

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
                loss = cdd_loss + trg_ent + self.hparams['TOV_wt'] * tov_loss + recons

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item(), 'recons_loss': recons.detach().item(), 'cdd_loss': cdd_loss.detach().item()}
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
    
    def collect_samples(self, loader):
        """
        收集样本的特征、标签和路径信息。

        参数:
            net (nn.Module): 深度学习模型。
            loader (DataLoader): 数据加载器。
        """
        samples = {}
        data_feat, data_gt, data_all, labels = [], [], [], []
        for x, gt, _ in iter(loader):
            data = x.float().to(self.device)
            data_all += [data]
            data_gt += [gt.long().to(self.device)]
            feature = self.feature_extractor.forward(data)[0]
            data_feat += [feature]
            label = self.classifier.forward(feature).argmax(dim=1)
            labels += [label]
        samples['data'] = torch.cat(data_all, dim=0).float()
        samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        samples['feature'] = torch.cat(data_feat, dim=0)
        samples['label'] = torch.cat(labels, dim=0)
        return samples
    
    def construct_surrogate_feature_sampler(self, samples, filtered_classes):
        """构建高斯采样器"""
        feats = samples['feature']
        # 这里的 label 来自聚类的伪标签 results
        plabels = self.clustering.samples['label'] 
        
        variance_mult = self.hparams.get('variance_mult', 1.0)
        
        # 计算 Target 的均值和方差
        target_mean = torch.zeros(self.configs.num_classes, feats.shape[-1])
        target_std = torch.zeros(self.configs.num_classes, feats.shape[-1])

        for c in filtered_classes:
            idx = torch.where(plabels == c)[0]
            if len(idx) > 1:
                _std, _mean = torch.std_mean(feats[idx], dim=0, unbiased=True)
                target_mean[c] = _mean
                target_std[c] = _std

        # 构建采样器
        for i in range(self.configs.num_classes):
            cur_target_norm = target_mean[i].norm(p=2)
            if cur_target_norm < 1e-3:
                self.surrogate_feature_sampler[i] = None
                continue
            
            # 【核心逻辑】
            # Mean: 方向取自源域 Anchor，长度取自目标域 (假设分布发生偏移但相对结构保持)
            estimated_source_mean = self.source_anchors[i] * cur_target_norm
            # Std: 使用目标域的 Std
            estimated_source_std = target_std[i]
            estimated_source_std[estimated_source_std == 0] = 1e-4
            
            self.surrogate_feature_sampler[i] = torch.distributions.normal.Normal(
                estimated_source_mean, estimated_source_std * variance_mult
            )

    def sample_source_features(self, nums_cls, num_classes):
        """根据需要的数量采样源域特征"""
        source_features = []
        feat_dim = self.configs.features_len * self.configs.final_out_channels + self.configs.fourier_modes*2/self.configs.stride
        for c in range(num_classes):
            num = nums_cls[c]
            if num > 0 and self.surrogate_feature_sampler.get(c) is not None:
                # 采样
                s = self.surrogate_feature_sampler[c].sample((num,))
                source_features.append(s)
            elif num > 0:
                # 如果该类无法采样（例如目标域该类样本太少），补零或者随机噪音
                source_features.append(torch.zeros(num, int(feat_dim)))
        
        if len(source_features) > 0:
            return torch.cat(source_features, dim=0)
        else:
            # 极少数情况 Batch 为空
            return torch.tensor([]).to(self.device)