import os
import urllib
import tarfile

from utilities.datareader import pickle_data

__author__ = 'uyaseen'


def fetch_data(b_path, _file):
    dataset_url = 'http://cs.stanford.edu/~danqi/data/' + _file
    print('downloading data from %s' % dataset_url)
    if not os.path.exists(b_path):
        os.makedirs(b_path)
    urllib.urlretrieve(dataset_url, b_path + _file)
    with tarfile.open(b_path + _file, 'r|gz') as t:
        t.extractall(b_path)


if __name__ == '__main__':
    task = 'cnn'  # let CNN be default dataset
    _b_dir = 'data/'
    _dir = _b_dir + task + '/raw/'
    if task == 'cnn':
        __file = 'cnn.tar.gz'
    elif task == 'dailymail':
        __file = 'dailymail.tar.gz'
    else:
        print('Only `cnn` & `dailymail` datasets are supported.')
    if not os.path.isfile(_dir + __file):
        fetch_data(_dir, __file)
    _d_path = _dir + task + '/'
    _w_path = _b_dir + task + '/'
    pickle_data(path=_d_path, w_path=_w_path, task=task)
    print('... done')
