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
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.91094, 'im': 0.98305,
                     'target_cls_wt': 0},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'beta': 5, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.91094, 'im': 0.98305,  'TOV_wt': 0.169},
            'SAFA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.91094, 'im': 0.98305,  'TOV_wt': 0.169, 'amp_scale': 0.23958, 'phase_scale': 0.18379,},
            'SF_FPA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.91094, 'im': 0.98305, 'align_wt': 0.037414, 'amp_scale': 0.23958, 'phase_scale': 0.18379, 'conf_thresh': 0.77752},
            'TemSR': {
                     'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.91094,
                     'disc_loss_wt': 50, 'src_like_entropy_wt': 3.45814, 'Gent': False, 'src_like_epochs': 3,
                     'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 5.26417,'AR_hid_dim': 128, 'num_splits': 8,
                     'num_masked': 1,'anchor_percent': 0.3, 'CL_temp': 0.05, 'detach': False
                     },
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.00001, 'ent_loss_wt': 0.91094, 'im': 0.98305,
                     'target_cls_wt': 1, 'aad_wt':0.01, 'beta': 10, 'alpha': 1},
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
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.91094, 'im': 0.10907,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.003, 'learning_rate': 0.001, 'beta': 10, 'alpha': 1},

            'NRC': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.41939, 'im': 0.10907,  'TOV_wt': 0.5},
            'SFTSDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.41939, 'im': 0.10907,  'TOV_wt': 0.5},
            'SAFA': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.41939, 'im': 0.10907,  'TOV_wt': 0.5, 'amp_scale': 0.59687, 'phase_scale': 0.57247,},
            'SF_FPA': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.41939, 'im': 0.10907, 'align_wt': 1.07693, 'amp_scale': 0.59687, 'phase_scale': 0.57247, 'conf_thresh': 0.90212},
            'TemSR': {
                     'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.41939,
                     'disc_loss_wt': 50, 'src_like_entropy_wt': 3.45814, 'Gent': False, 'src_like_epochs': 3,
                     'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 5.26417,'AR_hid_dim': 128, 'num_splits': 8,
                     'num_masked': 1,'anchor_percent': 0.3, 'CL_temp': 0.05, 'detach': False
                     },
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.001, 'ent_loss_wt': 0.41939, 'im': 0.10907,
                     'target_cls_wt': 0.15, 'aad_wt':0.001, 'beta': 9, 'alpha': 1},
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
            'SHOT': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.040631, 'im': 0.92552,
                     'target_cls_wt': 0.3312},
            'AaD': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'beta': 10, 'alpha': 1},
            'NRC': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'epsilon': 1e-5},
            'MAPU': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.040631, 'im': 0.92552,  'TOV_wt': 0.5},
            'SAFA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.040631, 'im': 0.92552,  'TOV_wt': 0.5, 'amp_scale': 0.7167, 'phase_scale': 0.46981},
            'SF_FPA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.040631, 'im': 0.92552, 'align_wt': 0.89126, 'amp_scale': 0.7167, 'phase_scale': 0.46981, 'conf_thresh': 0.64167},
            'TemSR': {
                     'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.040631,
                     'disc_loss_wt': 50, 'src_like_entropy_wt': 3.45814, 'Gent': False, 'src_like_epochs': 3,
                     'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 5.26417,'AR_hid_dim': 128, 'num_splits': 4,
                     'num_masked': 1,'anchor_percent': 0.3, 'CL_temp': 0.05, 'detach': False
                     },
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.040631, 'im': 0.92552,
                     'target_cls_wt': 0.15, 'aad_wt':0.001, 'beta': 9, 'alpha': 1},
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
            'SAFA': {'pre_learning_rate':  0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'TOV_wt': 0.6385, 'amp_scale': 0.76704, 'phase_scale': 0.88275},
            'SF_FPA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514, 'align_wt': 0.6385, 'amp_scale': 0.76704, 'phase_scale': 0.88275, 'conf_thresh': 0.7},
            'TemSR': {
                     'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 1,
                     'disc_loss_wt': 50, 'src_like_entropy_wt': 1, 'Gent': True, 'src_like_epochs': 3,
                     'trg_disc_epochs': 3,'trg_max_min_loss_wt': 2.09309,'AR_hid_dim': 64, 'num_splits': 8,
                     'num_masked': 1, 'anchor_percent': 0.3, 'CL_temp': 0.05, 'detach': True
                     },
            'CESFDA': {'pre_learning_rate': 0.003, 'learning_rate': 0.00001, 'ent_loss_wt': 0.4216, 'im': 0.5514,
                     'target_cls_wt': 0.15, 'aad_wt':0.001, 'beta': 9, 'alpha': 1},
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
            'TemSR':  {
                      'pre_learning_rate': 0.001, 'learning_rate': 0.0005, 'ent_loss_wt': 0.1,
                      'disc_loss_wt': 50, 'src_like_entropy_wt': 1.5, 'Gent': True, 'src_like_epochs': 3,
                      'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 3.64, 'AR_hid_dim': 64, 'num_splits': 8,
                      'num_masked': 1, 'anchor_percent':0.3, 'CL_temp': 0.01, 'detach': True
                      },
            'SAFA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, 'amp_scale': 0.76704, 'phase_scale': 0.88275,},
            'SF_FPA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759, 'align_wt': 0.5, 'amp_scale': 0.76704, 'phase_scale': 0.88275, 'conf_thresh': 0.7},
            'CESFDA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01,  'beta': 10, 'alpha': 1, 'aad_wt':0.025},
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
            'TemSR':  {
                      'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'ent_loss_wt': 0.1,
                      'disc_loss_wt': 50, 'src_like_entropy_wt': 1.5, 'Gent': True, 'src_like_epochs': 3,
                      'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 3.64, 'AR_hid_dim': 64, 'num_splits': 8,
                      'num_masked': 1, 'anchor_percent':0.3, 'CL_temp': 0.01, 'detach': True
                      },
            'CESFDA': {'pre_learning_rate': 0.003, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,
                     'target_cls_wt': 0.01, 'aad_wt':0.02,'beta': 9, 'alpha': 1},
            'SAFA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759,  'TOV_wt': 0.5, 'amp_scale': 0.76704, 'phase_scale': 0.88275,},
            'SF_FPA': {'pre_learning_rate': 0.001, 'learning_rate': 0.0001, 'ent_loss_wt': 0.05897, 'im': 0.2759, 'align_wt': 0.5, 'amp_scale': 0.76704, 'phase_scale': 0.88275, 'conf_thresh': 0.7},
        }