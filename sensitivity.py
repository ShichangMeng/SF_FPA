"""
敏感性分析：SF_FPA @ WISDM，单变量扫描 amp_scale / phase_scale。

运行方式（必须在仓库根目录执行，依赖相对路径 dataset/ 与 exp 输出目录）:
    python sensitivity.py                          # 两个参数都跑（默认）
    python sensitivity.py --param amp_scale        # 只跑 amp_scale
    python sensitivity.py --param phase_scale --num_runs 3
    python sensitivity.py --param amp_scale --values 0.2 0.4   # 自定义网格
    python sensitivity.py --plot_only              # 只根据已有 summary.csv 重画图

说明:
    - 网格默认取自 example _simple.ipynb (cell 18/19)，与 configs/sweep_params.py
      的搜索范围 (amp 0~1, phase 0~2.5) 一致。
    - 扫描某个参数时，另一个参数保持 configs/hparams.py 中 WISDM/SF_FPA 的调优值
      (amp_scale=0.59687, phase_scale=0.57247)。
    - 已存在 results.csv 的取值会自动跳过（断点续跑），--force 可强制重跑。

输出:
    exp0922_sens/WISDM/<param>_<value>/results.csv    每个取值的原始结果 (9 场景 x num_runs)
    exp0922_sens/summary.csv                          所有取值的均值汇总
    exp0922_sens/plots/MF1_<param>.{png,pdf}          MF1 曲线
"""
import sys
import os
import time
import argparse

sys.path.append('./')
sys.path.append('./trainers')

import pandas as pd
from train import Trainer  # trainers/train.py

# 单变量扫描网格（沿用 example _simple.ipynb 的先例）
DEFAULT_GRIDS = {
    'amp_scale': [0.2, 0.4, 0.6, 0.8, 1.0],
    'phase_scale': [0.2, 0.4, 0.6, 0.8, 1.0],
}

# configs/hparams.py -> WISDM / SF_FPA 调优默认值（扫描另一参数时固定在此）
FIXED_DEFAULTS = {
    'amp_scale': 0.59687,
    'phase_scale': 0.57247,
}

XLABELS = {
    'amp_scale': r'Amplitude Scale $\alpha$',
    'phase_scale': r'Phase Scale $\beta$',
}

SAVE_DIR = 'MAPU_SFDA_TS/exp1004_sens'


def build_args(run_description, num_runs, device):
    # 与 trainers/train.py __main__ 中的参数保持一致
    return argparse.Namespace(
        save_dir=SAVE_DIR,
        run_description=run_description,
        da_method='SF_FPA',
        data_path='dataset',
        dataset='WISDM',
        backbone='CNN',
        num_runs=num_runs,
        device=device,
    )


def summarize_results(results_csv):
    df = pd.read_csv(results_csv)
    clean = df[~df['scenario'].isin(['mean', 'std'])]
    return {
        'acc': clean['acc'].astype(float).mean(),
        'f1_score': clean['f1_score'].astype(float).mean(),
        'auroc': clean['auroc'].astype(float).mean(),
        'n_rows': len(clean),
    }


def run_one(param, value, num_runs, device, force):
    run_desc = f'{param}_{value:g}'
    args = build_args(run_desc, num_runs, device)
    trainer = Trainer(args)  # 每次重新构造，hparams 从 configs 重新加载

    # 覆盖被扫描参数；另一参数保持 hparams.py 默认值（显式写死，防配置漂移）
    trainer.hparams[param] = float(value)
    for k, v in FIXED_DEFAULTS.items():
        if k != param:
            trainer.hparams[k] = v

    results_csv = os.path.join(trainer.exp_log_dir, 'results.csv')
    if os.path.exists(results_csv) and not force:
        print(f'[{run_desc}] results.csv 已存在，跳过训练', flush=True)
        reused = True
    else:
        print(f'[{run_desc}] 开始训练: hparams amp_scale='
              f'{trainer.hparams["amp_scale"]}, phase_scale={trainer.hparams["phase_scale"]}',
              flush=True)
        t0 = time.time()
        trainer.train()
        print(f'[{run_desc}] 完成，用时 {time.time() - t0:.1f}s', flush=True)
        reused = False

    stats = summarize_results(results_csv)
    return {
        'param': param,
        'value': value,
        **stats,
        'reused': reused,
        'results': results_csv,
    }


def plot_summary(df, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    made = []
    for param, sub in df.groupby('param'):
        sub = sub.sort_values('value')
        fig, ax = plt.subplots(figsize=(3.8, 3.0), dpi=200)
        ax.plot(sub['value'], sub['f1_score'] * 100,
                marker='s', markersize=5, mfc='none',
                linewidth=1.25, color='#e47159', label='WISDM', linestyle='--')
        ax.set_xlabel(XLABELS.get(param, param), fontsize=14)
        ax.set_ylabel('MF1 Score ' + r'$\uparrow$', fontsize=14)
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.25)
        leg = ax.legend(frameon=True, fontsize=14, loc='best',
                        edgecolor='gray', fancybox=False)
        leg.get_frame().set_alpha(0.8)
        leg.get_frame().set_linewidth(0.5)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            out = os.path.join(plot_dir, f'MF1_{param}.{ext}')
            fig.savefig(out, dpi=300)
            made.append(out)
        plt.close(fig)
    return made


def main():
    p = argparse.ArgumentParser(description='SF_FPA/WISDM amp_scale & phase_scale 敏感性分析')
    p.add_argument('--param', choices=['amp_scale', 'phase_scale', 'both'], default='both')
    p.add_argument('--values', nargs='+', type=float, default=None,
                   help='自定义扫描网格（仅支持 --param 指定单个参数时使用）')
    p.add_argument('--num_runs', type=int, default=3, help='随机种子数，默认与主实验一致')
    p.add_argument('--device', default='cuda')
    p.add_argument('--force', action='store_true', help='忽略已有 results.csv 强制重跑')
    p.add_argument('--plot_only', action='store_true', help='只读取 summary.csv 重新画图')
    a = p.parse_args()

    # 关键路径依赖 cwd：AbstractTrainer 用 dirname(cwd) 拼输出目录，data_path 为相对路径
    assert os.path.exists('trainers/train.py') and os.path.exists('dataset/WISDM'), \
        '请在仓库根目录 (MAPU_SFDA_TS/) 下运行本脚本'

    home = os.path.dirname(os.getcwd())
    exp_root = os.path.join(home, SAVE_DIR)
    summary_csv = os.path.join(exp_root, 'summary.csv')
    plot_dir = os.path.join(exp_root, 'plots')
    os.makedirs(exp_root, exist_ok=True)

    if a.plot_only:
        df = pd.read_csv(summary_csv)
    else:
        params = list(DEFAULT_GRIDS) if a.param == 'both' else [a.param]
        if a.values is not None:
            assert len(params) == 1, '--values 只能配合单个 --param 使用'
            grids = {params[0]: a.values}
        else:
            grids = {k: DEFAULT_GRIDS[k] for k in params}

        rows = []
        if os.path.exists(summary_csv):
            rows = pd.read_csv(summary_csv).to_dict('records')

        for param in params:
            for value in grids[param]:
                # 断点续跑：summary 中已有该点且 results.csv 仍在则不重跑
                existing = [r for r in rows
                            if r['param'] == param and float(r['value']) == float(value)]
                results_csv = os.path.join(exp_root, 'WISDM', f'{param}_{value:g}', 'results.csv')
                if existing and os.path.exists(results_csv) and not a.force:
                    print(f'[{param}={value:g}] summary 已有，跳过', flush=True)
                    continue
                row = run_one(param, value, a.num_runs, a.device, a.force)
                rows = [r for r in rows
                        if not (r['param'] == param and float(r['value']) == float(value))]
                rows.append(row)
                # 每个点跑完立即落盘，中断不丢结果
                out_df = pd.DataFrame(rows).sort_values(['param', 'value'])
                out_df.to_csv(summary_csv, index=False)
                print(f'>>> 当前汇总:\n{out_df[["param", "value", "acc", "f1_score", "auroc"]]}',
                      flush=True)

        df = pd.DataFrame(rows).sort_values(['param', 'value'])

    df.to_csv(summary_csv, index=False)
    made = plot_summary(df, plot_dir)

    print('\n=== 敏感性分析结果 (MF1 = f1_score x 100) ===')
    pivot = df.copy()
    pivot['MF1'] = (pivot['f1_score'] * 100).round(3)
    print(pivot[['param', 'value', 'acc', 'MF1', 'auroc']].to_string(index=False))
    print(f'\nsummary: {summary_csv}')
    for m in made:
        print(f'plot:    {m}')


if __name__ == '__main__':
    main()
