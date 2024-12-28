import argparse
import numpy as np
import pickle

from PIL import Image
from torch.utils.data import Dataset

SERVER_ADDR= 'localhost'   # When running in a real distributed setting, change to the server's IP address
import torchvision.transforms as transforms

# SERVER_ADDR = '192.168.1.100'  # When running in a real distributed setting, change to the server's IP address
# SERVER_ADDR = '169.254.247.194'
SERVER_PORT = 51002


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_clients',
                        help='total number of clients',
                        type=int,
                        default=4)

    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=4)

    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=500)

    parser.add_argument('--E',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=50)

    parser.add_argument('--n',
                        help='data samples;',
                        type=int,
                        default=200)

    parser.add_argument('--lr',
                        help='learning rate;',
                        type=float,
                        default=0.5)

    parser.add_argument('--decay',
                        help='decay for the learning rate;',
                        type=float,
                        default=0.995)

    # is non-iid
    parser.add_argument('--iid',
                        help='is iid',
                        type=bool,
                        default=False)

    parser.add_argument('--data_set',
                        help='type of dataset',
                        type=str,
                        default='mnist')

    parser.add_argument('--non_iid_degree',
                        help='degree of non-iid',
                        type=int,
                        default=1)

    parser.add_argument('--num_straggle',
                        help='number of straggler',
                        type=int,
                        default=0)

    parser.add_argument('--sleep_secs',
                        help='sleep secs for straggler',
                        type=int,
                        default=10)

    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parsed = parser.parse_args()
    options = parsed.__dict__
    return options


def read_data(data_dir):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    # clients = []
    # groups = []
    data = {}
    print('>>> Read data from:', data_dir)

    # open training dataset pkl files
    with open(data_dir, 'rb') as inf:
        cdata = pickle.load(inf)

    data.update(cdata)
    data = MiniDataset(data['x'], data['y'])

    return data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target
