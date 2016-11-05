from __future__ import division

import cPickle as pkl
import os.path

import numpy as np
from sklearn.metrics import f1_score

import theano
import theano.tensor as T

from model.layers.gru import DeepGru, BiGru
from model.layers.lstm import DeepLstm, BiLstm
from model.layers.mlp import MLPAttn, LogisticRegression

from text_utils import tokenize
from utilities.loaddata import shared_data

__author__ = 'uyaseen'


def evaluate_attentive(m_path, test_data, vocabulary, emb_dim, hidden_dim, attn_dim,
                       model):
    assert os.path.isfile(m_path), True
    test_s, test_q, test_a = shared_data(test_data)
    if model is None:
        print('Please specify the model ! {attn-gru, attn-lstm}')
        return

    vocab, words_to_ix, ix_to_words = vocabulary
    with open(m_path, 'rb') as f:
        s_params, q_params, mlp_attn_params = pkl.load(f)  # model params
    # allocate symbolic variables for the story, question & answer
    s = T.imatrix('s')
    q = T.imatrix('q')
    a = T.ivector('a')
    vocab_size = len(vocab)
    if model == 'attn-gru':
        model_s = BiGru(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                        hidden_dim=hidden_dim, params=s_params)
        model_q = BiGru(input=q, vocab_size=vocab_size, emb_dim=emb_dim,
                        hidden_dim=hidden_dim, params=q_params)
    elif model == 'attn-lstm':
        model_s = BiLstm(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=s_params)
        model_q = BiLstm(input=q, vocab_size=vocab_size, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=q_params)
    else:
        print('Only bi-directional attention models (attn-gru, attn-lstm) are supported.')
        raise TypeError

    # story is encoded as concatenation of forward + backward gru/lstm at 'all' time-steps i.e input tokens
    story = T.concatenate([model_s.h_f.dimshuffle(1, 0, 2), model_s.h_b.dimshuffle(1, 0, 2)[::, ::-1]], axis=2)
    # Note: question = concatenation of hidden representation of forward + backward gru/lstm at 'last' time-step
    mlp_attn = MLPAttn(story=story, question=model_q.y, hidden_dim=hidden_dim * 2,
                       output_dim=vocab_size, attn_dim=attn_dim, params=mlp_attn_params)
    get_pred = theano.function(
        inputs=[],
        outputs=mlp_attn.pred,
        givens={
            s: test_s,
            q: test_q
        }
    )
    get_errors = theano.function(
        inputs=[],
        outputs=mlp_attn.errors(a),
        givens={
            s: test_s,
            q: test_q,
            a: test_a
        }
    )
    print('model = %s' % model)
    N = len(test_data[0])
    accuracy = (N -
                np.sum(get_errors())) / N * 100.
    y_true = test_data[4]
    y_pred = get_pred()
    score = f1_score(y_true=y_true, y_pred=y_pred,
                     average='weighted')
    print('accuracy: %.2f' % accuracy)
    print('F1-score: %.4f' % score)


def evaluate_deep(m_path, test_data, vocabulary, emb_dim, hidden_dim,
                  model):
    assert os.path.isfile(m_path), True
    test_s, test_q, test_a = shared_data(test_data)
    test_s.set_value(np.hstack([test_s.get_value(), test_q.get_value()]))
    if model is None:
        print('Please specify the model ! {deep-gru, deep-lstm}')
        return

    vocab, words_to_ix, ix_to_words = vocabulary
    with open(m_path, 'rb') as f:
        enc_params, logreg_params = pkl.load(f)
    # allocate symbolic variables for the story, question & answer
    s = T.imatrix('s')  # represents concatenated story + question
    a = T.ivector('a')
    vocab_size = len(vocab)
    if model == 'deep-gru':
        encoder = DeepGru(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                          hidden_dim=hidden_dim, n_layers=2, params=enc_params)
    elif model == 'deep-lstm':
        encoder = DeepLstm(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, n_layers=2, params=enc_params)
    else:
        print('Only deep models (deep-gru, deep-lstm) are supported.')
        raise TypeError
    # pass hidden representation of 'story + question' to our 'minimalist classifier'
    logreg = LogisticRegression(input=encoder.y, input_dim=hidden_dim * 2, output_dim=vocab_size,
                                params=logreg_params)
    get_pred = theano.function(
        inputs=[],
        outputs=logreg.pred,
        givens={
            s: test_s
        }
    )
    get_errors = theano.function(
        inputs=[],
        outputs=logreg.errors(a),
        givens={
            s: test_s,
            a: test_a
        }
    )
    print('model = %s' % model)
    N = len(test_data[0])
    accuracy = (N -
                np.sum(get_errors())) / N * 100.
    y_true = test_data[4]
    y_pred = get_pred()
    score = f1_score(y_true=y_true, y_pred=y_pred,
                     average='weighted')
    print('accuracy: %.2f' % accuracy)
    print('F1-score: %.4f' % score)


def answer_attentive(m_path, vocabulary, emb_dim, hidden_dim, attn_dim, model):
    assert os.path.isfile(m_path), True
    if model is None:
        print('Please specify the model ! {attn-gru, attn-lstm}')
        return

    vocab, words_to_ix, ix_to_words = vocabulary
    with open(m_path, 'rb') as f:
        s_params, q_params, mlp_attn_params = pkl.load(f)  # model params
    # allocate symbolic variables for the story & question
    s = T.imatrix('s')
    q = T.imatrix('q')
    vocab_size = len(vocab)
    if model == 'attn-gru':
        model_s = BiGru(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                        hidden_dim=hidden_dim, params=s_params)
        model_q = BiGru(input=q, vocab_size=vocab_size, emb_dim=emb_dim,
                        hidden_dim=hidden_dim, params=q_params)
    elif model == 'attn-lstm':
        model_s = BiLstm(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=s_params)
        model_q = BiLstm(input=q, vocab_size=vocab_size, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, params=q_params)
    else:
        print('Only bi-directional attention models (attn-gru, attn-lstm) are supported.')
        raise TypeError

    # story is encoded as concatenation of forward + backward gru/lstm at 'all' time-steps i.e input tokens
    story = T.concatenate([model_s.h_f.dimshuffle(1, 0, 2), model_s.h_b.dimshuffle(1, 0, 2)[::, ::-1]], axis=2)
    # Note: question = concatenation of hidden representation of forward + backward gru/lstm at 'last' time-step
    mlp_attn = MLPAttn(story=story, question=model_q.y, hidden_dim=hidden_dim * 2,
                       output_dim=vocab_size, attn_dim=attn_dim, params=mlp_attn_params)
    get_pred = theano.function(
        inputs=[s, q],
        outputs=mlp_attn.pred
    )
    unk = 'UNKNOWN_TOKEN'
    while True:
        st = raw_input('Story >> ')
        ques = raw_input('Question >> ')
        st = [words_to_ix[wd] if wd in vocab else words_to_ix[unk]
              for wd in tokenize(st)]
        ques = [words_to_ix[wd] if wd in vocab else words_to_ix[unk]
                    for wd in tokenize(ques)]
        if len(st) and len(ques) > 0:
            story_ = [[]]
            story_[0] = st
            question = [[]]
            question[0] = ques
            ans = get_pred(story_, question)
            ans = ix_to_words[ans[0]]
            print('Answer >> ', ans)
        else:
            print('Story/Question should at-least be couple of words :/ ...')


def answer_deep(m_path, vocabulary, emb_dim, hidden_dim, model):
    assert os.path.isfile(m_path), True
    if model is None:
        print('Please specify the model ! {deep-gru, deep-lstm}')
        return

    vocab, words_to_ix, ix_to_words = vocabulary
    with open(m_path, 'rb') as f:
        enc_params, logreg_params = pkl.load(f)
    # allocate symbolic variables for the story | question
    s = T.imatrix('s')  # represents concatenated story + question
    vocab_size = len(vocab)
    if model == 'deep-gru':
        encoder = DeepGru(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                          hidden_dim=hidden_dim, n_layers=2, params=enc_params)
    elif model == 'deep-lstm':
        encoder = DeepLstm(input=s, vocab_size=vocab_size, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, n_layers=2, params=enc_params)
    else:
        print('Only deep models (deep-gru, deep-lstm) are supported.')
        raise TypeError
    # pass hidden representation of 'story + question' to our 'minimalist classifier'
    logreg = LogisticRegression(input=encoder.y, input_dim=hidden_dim * 2, output_dim=vocab_size,
                                params=logreg_params)
    get_pred = theano.function(
        inputs=[s],
        outputs=logreg.pred
    )
    unk = 'UNKNOWN_TOKEN'
    while True:
        story = raw_input('Story >> ')
        question = raw_input('Question >> ')
        story = [words_to_ix[wd] if wd in vocab else words_to_ix[unk]
                 for wd in tokenize(story)]
        question = [words_to_ix[wd] if wd in vocab else words_to_ix[unk]
                    for wd in tokenize(question)]
        if len(story) and len(question) > 0:
            story_ques = [[]]
            story_ques[0] = story + question
            ans = get_pred(story_ques)
            ans = ix_to_words[ans[0]]
            print('Answer >> ', ans)
        else:
            print('Story/Question should at-least be couple of words :/ ...')
