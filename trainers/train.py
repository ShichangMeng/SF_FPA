import sys

sys.path.append('./')

import os
import pandas as pd

import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def train(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Average meters
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)
                self.current_source_id, self.current_target_id, self.current_run_id = src_id, trg_id, run_id

                # Train model
                non_adapted_model, last_adapted_model, best_adapted_model = self.train_model()

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model, last_adapted_model,
                                     best_adapted_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)

        # Save tables to file
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='MAPU_SFDA_TS/exp1004', type=str,
                        help='Directory containing all experiments')
    parser.add_argument('--run_description', default=None, type=str, help='Description of run, if none, DA method name will be used')

    # ========= Select the DA methods ===========
    parser.add_argument('--da_method', default='SF_FPA', type=str, help='SHOT, AaD, NRC, MAPU, TemSR, CESFDA, SAFA')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default='dataset', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='WISDM', type=str, help='Dataset of choice: (WISDM - Boiler - FD)')

    # ========= Select the BACKBONE =============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings =============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--source_checkpoint_root', default=None, type=str,
                        help='Load each scenario/seed source model from this algorithm checkpoint directory and skip pretraining')
    parser.add_argument('--add_fal', action='store_true',
                        help='Add trainable FAL while freezing encoder weights; use the selected baseline update loss')

    args = parser.parse_args()
    if args.add_fal and not args.source_checkpoint_root:
        parser.error('--add_fal requires --source_checkpoint_root')
    if args.add_fal and args.da_method not in ('SHOT', 'AaD', 'NRC'):
        parser.error('--add_fal is currently implemented for SHOT, AaD, and NRC')

    trainer = Trainer(args)
    trainer.train()
