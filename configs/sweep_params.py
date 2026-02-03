
sweep_alg_hparams = {

    'MAPU_Anchor_freq_gmm':
        {   
            # 'amp_scale': {'distribution': 'uniform', 'min': 0, 'max': 1},
            # 'phase_scale': {'distribution': 'uniform', 'min': 0, 'max': 3},
            'amp_scale': {'values': [0.2, 0.4, 0.6, 0.8, 1.0]},
            # 'phase_scale': {'values': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
            # 'ent_loss_wt': {'values': [0.01, 0.1, 0.5, 0.8, 1.0]}, 
            # 'im':          {'values': [0.1, 0.3, 0.5, 0.7, 1.0]},
            # 'TOV_wt':      {'values': [0.1, 0.3, 0.5, 0.7, 1.0]},
        },
}
