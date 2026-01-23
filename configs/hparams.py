def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,
                     'target_cls_wt': 0},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'TF_MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169},
            'TF_MAPU_Anchor_sink': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169, 'sink_weight': 0.005, 'eps': 0.1},
            'TF_MAPU_Anchor': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
            'TF_MAPU_Anchor_freq': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.8467, 'im': 0.2983,  'TOV_wt': 0.169, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.0081},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'beta': 9, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'TF_MAPU': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385},
            'TF_MAPU_Anchor': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
            'TF_MAPU_Anchor_freq': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},

            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU_Anchor_sink': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, 'sink_weight': 0.005},
            'TF_MAPU_Anchor': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
            'TF_MAPU_Anchor_freq': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
        }

class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
                'num_epochs': 50,
                'batch_size': 64,
                'weight_decay': 1e-4,
                'step_size': 50,
                'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.001, 'beta': 10, 'alpha': 1},

            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU_Anchor_sink': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, 'sink_weight': 0.005},
            'TF_MAPU_Anchor': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
            'TF_MAPU_Anchor_freq': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
        }

class HHAR_SA():
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},

            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU_Anchor_sink': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, 'sink_weight': 0.005},
            'TF_MAPU_Anchor': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
            'TF_MAPU_Anchor_freq': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
        }

class Boiler():
    def __init__(self):
        super(Boiler, self).__init__()
        self.train_params = {
                'num_epochs': 30,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'step_size': 50,
                'lr_decay': 0.5
        }
        self.alg_hparams = {
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.6709, 'im': 0.8969,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5},
            'TF_MAPU_Anchor_sink': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, 'sink_weight': 0.005},
            'TF_MAPU_Anchor': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
            'TF_MAPU_Anchor_freq': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, "kernel_num": [5, 5], "kernel_mul": [2.0, 2.0], "cdd_weight": 0.3, "variance_mult": 1.0,},
        }