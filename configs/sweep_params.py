
sweep_alg_hparams = {

    'MAPU_Anchor_freq_gmm':
        {   
            'amp_scale': {'distribution': 'uniform', 'min': 0.2, 'max': 1},
            'phase_scale': {'distribution': 'uniform', 'min': 0.2, 'max': 1},
            # 'ent_loss_wt': {'values': [0.01, 0.1, 0.5, 0.8, 1.0]}, 
            # 'im':          {'values': [0.1, 0.3, 0.5, 0.7, 1.0]},
            # 'TOV_wt':      {'values': [0.1, 0.3, 0.5, 0.7, 1.0]},
        },
}
