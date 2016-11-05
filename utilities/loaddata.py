import theano
import numpy as np


__author__ = 'uyaseen'


def shared_data(data_xy, borrow=True):
    # ignoring 'masks' for story & question as they are not crucial for `many-to-one` sequence modelling
    data_s, _, data_q, _, data_a = data_xy
    shared_s = theano.shared(np.asarray(data_s,
                                        dtype='int32'),
                             borrow=borrow)
    shared_q = theano.shared(np.asarray(data_q,
                                        dtype='int32'),
                             borrow=borrow)
    shared_a = theano.shared(np.asarray(data_a,
                                        dtype='int32'),
                             borrow=borrow)

    return shared_s, shared_q, shared_a


def load_data(dataset):
    train_set, valid_set, test_set = dataset

    print('... transferring data to the %s' % theano.config.device)
    train_set_s, train_set_q, train_set_a = shared_data(train_set)
    valid_set_s, valid_set_q, valid_set_a = shared_data(valid_set)
    test_set_s, test_set_q, test_set_a = shared_data(test_set)

    return [[train_set_s, train_set_q, train_set_a],
            [valid_set_s, valid_set_q, valid_set_a],
            [test_set_s, test_set_q, test_set_a]]
