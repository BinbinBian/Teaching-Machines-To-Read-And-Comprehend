from __future__ import division
import timeit
import os.path
import cPickle as pkl
import numpy as np

from sklearn.metrics import f1_score
import theano
import theano.tensor as T

from layers.gru import DeepGru
from layers.lstm import DeepLstm
from layers.mlp import LogisticRegression
from utilities.optimizers import get_optimizer
from utilities.loaddata import load_data

__author__ = 'uyaseen'


def train_deep(dataset, vocabulary, b_path, task, rec_model='deep-gru', emb_dim=100, hidden_dim=256,
               use_existing_model=True, optimizer='rmsprop', n_epochs=50, batch_size=30):
    print('... train_deep(..)')
    vocab, words_to_ix, ix_to_words = vocabulary
    train, valid, test = load_data(dataset)
    train_set_s, train_set_q, train_set_a = train
    valid_set_s, valid_set_q, valid_set_a = valid
    test_set_s, test_set_q, test_set_a = test
    # concatenating story & question
    train_set_s.set_value(np.hstack([train_set_s.get_value(borrow=True),
                                     train_set_q.get_value(borrow=True)]))
    valid_set_s.set_value(np.hstack([valid_set_s.get_value(borrow=True),
                                     valid_set_q.get_value(borrow=True)]))
    test_set_s.set_value(np.hstack([test_set_s.get_value(borrow=True),
                                   test_set_q.get_value(borrow=True)]))
    n_train_batches = int(train_set_s.get_value(borrow=True).shape[0] / batch_size)
    print('... building the model')
    # allocate symbolic variables for the 'story + question' & 'answer'
    s = T.imatrix('s')  # represents concatenated story + question
    a = T.ivector('a')
    index = T.lscalar('index')
    vocab_size = len(vocab)
    m_path = b_path + 'models/' + task + '_' + rec_model + '_best_model.pkl'

    enc_params = None
    logreg_params = None
    if not os.path.exists(b_path + 'models/'):
        os.makedirs(b_path + 'models/')
    if use_existing_model:
        if rec_model == 'deep-gru' or rec_model == 'deep-lstm':
            if os.path.isfile(m_path):
                with open(m_path, 'rb') as f:
                    enc_params, logreg_params = pkl.load(f)
            else:
                print('Unable to load existing model "%s" , initializing model with random weights' % m_path)
        else:
            print('Only deep models (deep-gru, deep-lstm) are supported.')
            raise TypeError

    if rec_model == 'deep-gru':
        encoder = DeepGru(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                          hidden_dim=hidden_dim, n_layers=2, params=enc_params)
    elif rec_model == 'deep-lstm':
        encoder = DeepLstm(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, n_layers=2, params=enc_params)
    else:
        print('Only deep models (deep-gru, deep-lstm) are supported.')
        raise TypeError
    # pass hidden representation of 'story + question' to our 'minimalist classifier'
    logreg = LogisticRegression(input=encoder.y, input_dim=hidden_dim * 2, output_dim=vocab_size,
                                params=logreg_params)
    all_params = encoder.params + logreg.params
    cost = logreg.negative_log_likelihood(a)

    updates = get_optimizer(optimizer, cost, all_params)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            s: train_set_s[index * batch_size: (index + 1) * batch_size],
            a: train_set_a[index * batch_size: (index + 1) * batch_size]
        },
        updates=updates
    )
    get_errors = theano.function(
        inputs=[s, a],
        outputs=logreg.errors(a)
    )
    get_valid_pred = theano.function(
        inputs=[],
        outputs=logreg.pred,
        givens={
            s: valid_set_s
        }
    )
    print('... model: %s' % rec_model)
    print('... training')
    n_train_examples = train_set_s.get_value(borrow=True).shape[0]
    n_valid_examples = valid_set_s.get_value(borrow=True).shape[0]
    n_test_examples = test_set_s.get_value(borrow=True).shape[0]
    best_valid_score = -np.inf
    validation_freq = 1  # check the 'F1-score' after going through these many 'epochs'
    epoch = 0
    start_time = timeit.default_timer()
    while epoch < n_epochs:
        epoch += 1
        ep_start_time = timeit.default_timer()
        train_cost = 0.
        for i in xrange(n_train_batches):
            train_cost += train_model(i)

        if epoch % validation_freq == 0:
            ep_end_time = (timeit.default_timer() - ep_start_time) / 60.

            y_true = dataset[1][4]
            y_pred = get_valid_pred()
            valid_score = f1_score(y_true=y_true, y_pred=y_pred,
                                   average='weighted')

            epoch_log = 'epoch %i/%i, cross-entropy error: %f, ' \
                        'validation F1-score: %.4f, /epoch: %.4fm' % \
                        (epoch, n_epochs, train_cost/n_train_batches,
                         valid_score, ep_end_time)
            print(epoch_log)
            # save the current best model
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                with open(m_path, 'wb') as f:
                    dump_params = encoder.params, logreg.params
                    pkl.dump(dump_params, f, pkl.HIGHEST_PROTOCOL)

    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    print('--Final Metrics--')

    tr_accuracy = (n_train_examples -
                   np.sum(get_errors(np.asarray(np.hstack([dataset[0][0], dataset[0][2]]), dtype='int32'),
                                     np.asarray(dataset[0][4], dtype='int32')))) / n_train_examples * 100.
    va_accuracy = (n_valid_examples -
                   np.sum(get_errors(np.asarray(np.hstack([dataset[1][0], dataset[1][2]]), dtype='int32'),
                                     np.asarray(dataset[1][4], dtype='int32')))) / n_valid_examples * 100.
    te_accuracy = (n_test_examples -
                   np.sum(get_errors(np.asarray(np.hstack([dataset[2][0], dataset[2][2]]), dtype='int32'),
                                     np.asarray(dataset[2][4], dtype='int32')))) / n_test_examples * 100.
    print('train accuracy: ', tr_accuracy)
    print('validation accuracy: ', va_accuracy)
    print('test accuracy: ', te_accuracy)
