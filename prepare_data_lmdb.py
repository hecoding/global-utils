import argparse
import lmdb
from io import BytesIO
from tqdm import tqdm
from torchvision import datasets


def to_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=100)
    return buffer.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Input folder')
    parser.add_argument('--out', type=str, default='lmdb_out', help='Output folder')
    args = parser.parse_args()

    dataset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            for i, (img, label) in enumerate(tqdm(dataset)):
                key = str(i).encode('utf-8')
                txn.put(key, to_bytes(img))

                key = f'label-{str(i)}'.encode('utf-8')
                txn.put(key, str(label).encode('utf-8'))

            txn.put('length'.encode('utf-8'), str(len(dataset)).encode('utf-8'))
