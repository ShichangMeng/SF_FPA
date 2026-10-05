import argparse
import json
import sys

sys.path.append('.')
sys.path.append('trainers')
from types import SimpleNamespace

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


def make_args(args):
    return SimpleNamespace(
        save_dir=args.save_dir,
        run_description=args.run_description,
        da_method='SF_FPA',
        data_path='dataset',
        dataset='EEG',
        backbone='CNN',
        num_runs=args.num_runs,
        device=args.device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--run_description', required=True)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overrides', default='')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    print('OVERRIDES', json.dumps(overrides, sort_keys=True))

    if args.mode in ['train', 'both']:
        trainer = Trainer(make_args(args))
        trainer.hparams.update(overrides)
        trainer.train()

    if args.mode in ['test', 'both']:
        tester = TargetTest(make_args(args))
        tester.hparams.update(overrides)
        tester.scenario_test()


if __name__ == '__main__':
    main()
