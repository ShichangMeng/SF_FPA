import os
import torch


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 3
        self.dropout = 0.2

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 25 # for my model
        self.AR_hid_dim = 8

        # AR Discriminator
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1

        self.fourier_modes = 300
        self.intra_only = True
        self.num_splits = 8

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        # self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
        self.scenarios = [("0", "1"), ("0", "2"), ("0", "3"), ("1", "0"), ("1", "3"), ("2", "0"), ("2", "1"), ("3", "0"), ("3", "2")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 1
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.AR_hid_dim = 128
        self.fourier_modes = 64
        
        self.intra_only = True
        self.num_splits = 8

        self.AR_hid_dim_raw = self.input_channels

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16"),  ]
        # self.scenarios = [("7", "13")]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # self.features_len = 18 for sequential methods
        self.AR_hid_dim = 128
        self.fourier_modes = 64
        
        self.intra_only = False
        self.num_splits = 8

class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        # Closed Set DA
        #[("2", "32"), ("7", "30"), ('18','20'), ('20','30'), ('26','2')]
        self.scenarios = [("2", "32"), ("7", "30"), ('18','20'), ('20','30'), ('26','2'), ('9','3'), ('22','18'), ('12','9'), ('2','20')]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.width = 64  # for FNN
        self.fourier_modes = 64
        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.out_dim = 192
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        
        self.AR_hid_dim = 128

        self.intra_only = False
        self.num_splits = 8
        self.AR_hid_dim_raw = self.input_channels

class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        # self.scenarios = [("0", "2")]
        self.scenarios = [("0", "2"), ("1", "6"),("2", "4"),("4", "0"),("4", "5"),\
            ("5", "1"),("5", "2"),("7", "2"),("7", "5"),("8", "4")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 64
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64 * 2
        self.final_out_channels = 128
        self.features_len = 1
        self.out_dim = 128

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

        self.AR_hid_dim = 128

        self.intra_only = True
        self.num_splits = 8

class Boiler(object):
    def __init__(self):
        super(Boiler, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 6
        # self.scenarios = [("1", "2"),("1", "3"),("2", "3")]
        self.scenarios = [("1", "3"),("2", "3")]
        self.num_classes = 2
        self.sequence_len = 36
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 20
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 32
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        self.fourier_modes = 18
        self.AR_hid_dim = 32
        self.intra_only = True
        self.num_splits = 4
        self.AR_hid_dim_raw = self.input_channels