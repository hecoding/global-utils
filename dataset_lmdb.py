from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, path, transform=None):
        self.env = lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            img_bytes = txn.get(key)

            key = f'label-{str(index)}'.encode('utf-8')
            label = txn.get(key)

        img = self.from_bytes(img_bytes)
        if self.transform is not None:
            img = self.transform(img)

        label = int(label.decode('utf-8'))

        return img, label

    @staticmethod
    def from_bytes(b):
        buffer = BytesIO(b)
        return Image.open(buffer)
