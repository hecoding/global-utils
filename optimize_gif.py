#! /usr/bin/env python
import argparse
from pygifsicle import optimize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read Tensorboard logs')
    parser.add_argument('file', type=str)
    parser.add_argument('--destination', default=None, type=str, help='Path where to save updated gif. By default the old image is overwrited.')
    args = parser.parse_args()

    optimize(args.file, destination=args.destination)

