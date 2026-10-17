import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import pandas as pd
import warnings
import sklearn.exceptions
import time
from pathlib import Path
from copy import deepcopy
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class, FrequencyAdaptationLayer_Independent
from torch import nn

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """Shared configuration, training, evaluation, and result utilities."""

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.run_description = args.run_description if args.run_description is not None else args.da_method

        self.experiment_description = args.dataset


        # paths
        self.home_path = os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))


        # Specify runs
        self.num_runs = args.num_runs
        self.source_checkpoint_root = getattr(args, "source_checkpoint_root", None)
        self.add_fal = getattr(args, "add_fal", False)
        self.current_source_id = None
        self.current_target_id = None
        self.current_run_id = None

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        


    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        if self.source_checkpoint_root:
            checkpoint_path = Path(self.source_checkpoint_root) / (
                f"{self.current_source_id}_to_{self.current_target_id}_run_{self.current_run_id}"
            ) / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.algorithm.network.load_state_dict(checkpoint["non_adapted"], strict=True)
            self.non_adapted_model = checkpoint["non_adapted"]

            if self.add_fal:
                self.attach_fal(self.algorithm, freeze_encoder=True)
                trainable = [p for p in self.algorithm.parameters() if p.requires_grad]
                self.algorithm.optimizer = torch.optim.Adam(
                    trainable,
                    lr=self.hparams["learning_rate"],
                    weight_decay=self.hparams["weight_decay"],
                )
                self.non_adapted_model = deepcopy(self.algorithm.network.state_dict())
            self.logger.debug(f'Loaded source checkpoint: {checkpoint_path}')
        else:
            # pretraining step
            self.logger.debug(f'Pretraining stage..........')
            self.logger.debug("=" * 45)
            pretrain_start_time = time.time()
            self.non_adapted_model = self.algorithm.pretrain(self.src_train_dl, self.pre_loss_avg_meters, self.logger)
            pretrain_duration = time.time() - pretrain_start_time
            self.logger.debug(f'Pretraining stage finished in {pretrain_duration:.2f} seconds.')
        # adapting step
        self.logger.debug("=" * 45)
        self.logger.debug(f'Adaptation stage..........')
        self.logger.debug("=" * 45)
        adapt_start_time = time.time()
        self.last_model, self.best_model = self.algorithm.update(self.trg_train_dl, self.loss_avg_meters, self.logger)
        adapt_duration = time.time() - adapt_start_time
        self.logger.debug(f'Adaptation stage finished in {adapt_duration:.2f} seconds.')
        return  self.non_adapted_model,  self.last_model, self.best_model

    def attach_fal(self, algorithm, freeze_encoder=False):
        fal_hp = self.hparams_class.alg_hparams["SF_FPA"]
        algorithm.freq_adapter = FrequencyAdaptationLayer_Independent(
            seq_len=self.dataset_configs.sequence_len,
            channels=self.dataset_configs.input_channels,
            amp_scale=fal_hp["amp_scale"],
            phase_scale=fal_hp["phase_scale"],
            phase_modes=fal_hp.get("phase_modes"),
        ).to(self.device)
        if freeze_encoder:
            for parameter in algorithm.feature_extractor.parameters():
                parameter.requires_grad = False
        algorithm.network = nn.Sequential(
            algorithm.freq_adapter,
            algorithm.feature_extractor,
            algorithm.classifier,
        )
        return algorithm

    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)
        if hasattr(self.algorithm, 'freq_adapter'):
            freq_adapter = self.algorithm.freq_adapter.to(self.device)
            feature_extractor = nn.Sequential(freq_adapter, feature_extractor)
        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels,_ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)[0]
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')

        risks = src_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, non_adapted, last_model, best_model):
        save_dict = {
            "non_adapted":non_adapted,
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_metrics(self):
       
        self.evaluate(self.trg_test_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        return acc, f1, auroc

    def calculate_risks(self):
         # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table
