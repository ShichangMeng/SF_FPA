import argparse
import json
import sys
from types import SimpleNamespace

sys.path.append('.')
sys.path.append('trainers')

from trainers.train import Trainer
from trainers.test import TargetTest


def parse_overrides(text):
    overrides = {}
    if not text:
        return overrides
    for item in text.split(','):
        if not item.strip():
            continue
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        try:
            if any(ch in value for ch in ['.', 'e', 'E']):
                parsed = float(value)
            else:
                parsed = int(value)
        except ValueError:
            if value.lower() == 'true':
                parsed = True
            elif value.lower() == 'false':
                parsed = False
            else:
                parsed = value
        overrides[key] = parsed
    return overrides


def parse_scenarios(text):
    if not text:
        return None
    scenarios = []
    for item in text.split(','):
        src, trg = item.split('-', 1)
        scenarios.append((src.strip(), trg.strip()))
    return scenarios


def make_args(args):
    return SimpleNamespace(
        save_dir=args.save_dir,
        run_description=args.run_description,
        da_method='SF_FPA',
        data_path='dataset',
        dataset='Boiler',
        backbone='CNN',
        num_runs=args.num_runs,
        device=args.device,
    )


def apply_runtime_options(obj, hparam_overrides, scenarios):
    obj.hparams.update(hparam_overrides)
    if scenarios is not None:
        obj.dataset_configs.scenarios = scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--run_description', required=True)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overrides', default='')
    parser.add_argument('--scenarios', default='')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    scenarios = parse_scenarios(args.scenarios)
    print('OVERRIDES', json.dumps(overrides, sort_keys=True))
    print('SCENARIOS', scenarios)

    if args.mode in ['train', 'both']:
        trainer = Trainer(make_args(args))
        apply_runtime_options(trainer, overrides, scenarios)
        trainer.train()

    if args.mode in ['test', 'both']:
        tester = TargetTest(make_args(args))
        apply_runtime_options(tester, overrides, scenarios)
        tester.scenario_test()


if __name__ == '__main__':
    main()
