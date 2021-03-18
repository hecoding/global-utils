#! /usr/bin/env python
import argparse
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

LOAD_ALL = {'compressed_histograms': 0, 'images': 0, 'audio': 0, 'scalars': 0, 'histograms': 0, 'tensors': 0}


def read_scalars(path, load_all=False):
    path = Path(path)
    summaries = {}
    loading_sizes = None
    if load_all:
        loading_sizes = {'scalars': 0}

    for dir in sorted(path.iterdir()):
        ea = EventAccumulator(str(dir), size_guidance=loading_sizes).Reload()
        summaries[dir.name] = {'scalars': {}}
        for tag in ea.Tags()['scalars']:  # pd.DataFrame(ea.Scalars('loss/G'))
            summaries[dir.name]['scalars'][tag] = {}
            wall_time, summaries[dir.name]['scalars'][tag]['steps'], summaries[dir.name]['scalars'][tag]['values'] = zip(*ea.Scalars(tag))

    return summaries


def read_run_images(run_path, tag='images', load_all=False):
    loading_sizes = {'images': 50}
    if load_all:
        loading_sizes = {'images': 0}

    ea = EventAccumulator(run_path, size_guidance=loading_sizes).Reload()
    images = ea.Images(tag)

    """conda install imageio
    https://github.com/LucaCappelletti94/pygifsicle"""
    import imageio
    from pygifsicle import optimize
    out = 'elgif.gif'
    images_read = [imageio.imread(im.encoded_image_string) for im in images]
    imageio.mimsave(out, images_read, fps=10)
    optimize(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read Tensorboard logs')
    parser.add_argument('--logdir', required=True, type=str)
    parser.add_argument('--out', type=str, default='summaries.pickle', help='output file (default: summaries.pickle)')
    parser.add_argument('--load_all', action='store_true')
    args = parser.parse_args()

    summs = read_scalars(args.logdir, load_all=args.load_all)

    with open(args.out, 'wb') as f:
        pickle.dump(summs, f, protocol=4)  # pickle.HIGHEST_PROTOCOL)

    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)
