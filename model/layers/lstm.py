import theano
import theano.tensor as T

import numpy as np

from utilities.initializations import get

__author__ = 'uyaseen'


# Stack LSTM with skip-connections.
class DeepLstm(object):
    def __init__(self, input, vocab_size, emb_dim, hidden_dim, n_layers=2, init='uniform',
                 inner_init='orthonormal', inner_activation=T.nnet.hard_sigmoid,
                 activation=T.tanh, params=None):
        input = input.dimshuffle(1, 0)
        assert(n_layers == 2)  # can only stack one layer
        if params is None:
            self.emb = theano.shared(value=get(identifier=init, shape=(vocab_size, emb_dim), scale=np.sqrt(3)),
                                     name='emb', borrow=True)
            # *** Layer 1 ***
            # input gate
            self.W_i = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_i', borrow=True)
            self.U_i = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_i', borrow=True)
            self.b_i = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_i', borrow=True)
            # forget gate
            self.W_f = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_f', borrow=True)
            self.U_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_f', borrow=True)
            self.b_f = theano.shared(value=get(identifier='one', shape=(hidden_dim, )),
                                     name='b_f', borrow=True)
            # memory
            self.W_c = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_c', borrow=True)
            self.U_c = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_c', borrow=True)
            self.b_c = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_c', borrow=True)
            # output gate
            self.W_o = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_o', borrow=True)
            self.U_o = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_o', borrow=True)
            self.b_o = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                     name='b_o', borrow=True)

            # *** Layer 2 ***
            # input gate
            self.W_i_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_i_1', borrow=True)
            self.U_i_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_i_1', borrow=True)
            self.b_i_1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                       name='b_i_1', borrow=True)
            # forget gate
            self.W_f_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_f_1', borrow=True)
            self.U_f_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_f_1', borrow=True)
            self.b_f_1 = theano.shared(value=get(identifier='one', shape=(hidden_dim, )),
                                       name='b_f_1', borrow=True)
            # memory
            self.W_c_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_c_1', borrow=True)
            self.U_c_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_c_1', borrow=True)
            self.b_c_1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                       name='b_c_1', borrow=True)
            # output gate
            self.W_o_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_o_1', borrow=True)
            self.U_o_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_o_1', borrow=True)
            self.b_o_1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                       name='b_o_1', borrow=True)

        else:
            self.emb, self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, self.b_f, \
                self.W_c, self.U_c, self.b_c, self.W_o, self.U_o, self.b_o, \
                self.W_i_1, self.U_i_1, self.b_i_1, self.W_f_1, self.U_f_1, self.b_f_1, \
                self.W_c_1, self.U_c_1, self.b_c_1, self.W_o_1, self.U_o_1, self.b_o_1 = params

        self.c0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='c0', borrow=True)
        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.c1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='c1', borrow=True)
        self.h1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h1', borrow=True)
        self.params = [self.emb,
                       self.W_i, self.U_i, self.b_i,
                       self.W_f, self.U_f, self.b_f,
                       self.W_c, self.U_c, self.b_c,
                       self.W_o, self.U_o, self.b_o,
                       self.W_i_1, self.U_i_1, self.b_i_1,
                       self.W_f_1, self.U_f_1, self.b_f_1,
                       self.W_c_1, self.U_c_1, self.b_c_1,
                       self.W_o_1, self.U_o_1, self.b_o_1]

        def recurrence(x_t, c_t1_prev, h_t1_prev, c_t2_prev, h_t2_prev):
            # Layer 1 computation
            x_i = T.dot(self.emb[x_t], self.W_i) + self.b_i
            x_f = T.dot(self.emb[x_t], self.W_f) + self.b_f
            x_c = T.dot(self.emb[x_t], self.W_c) + self.b_c
            x_o = T.dot(self.emb[x_t], self.W_o) + self.b_o

            i_t = inner_activation(x_i + T.dot(h_t1_prev, self.U_i))
            f_t = inner_activation(x_f + T.dot(h_t1_prev, self.U_f))
            c_t = f_t * c_t1_prev + i_t * activation(x_c + T.dot(h_t1_prev, self.U_c))  # internal memory
            o_t = inner_activation(x_o + T.dot(h_t1_prev, self.U_o))
            h_t = o_t * activation(c_t)  # actual hidden state

            # Layer 2 computation
            x_i_1 = T.dot(h_t, self.W_i_1) + self.b_i_1
            x_f_1 = T.dot(h_t, self.W_f_1) + self.b_f_1
            x_c_1 = T.dot(h_t, self.W_c_1) + self.b_c_1
            x_o_1 = T.dot(h_t, self.W_o_1) + self.b_o_1

            i_t_1 = inner_activation(x_i_1 + T.dot(h_t2_prev, self.U_i_1))
            f_t_1 = inner_activation(x_f_1 + T.dot(h_t2_prev, self.U_f_1))
            c_t_1 = f_t_1 * c_t2_prev + i_t_1 * activation(x_c_1 + T.dot(h_t2_prev, self.U_c_1))  # internal memory
            o_t_1 = inner_activation(x_o_1 + T.dot(h_t2_prev, self.U_o_1))
            h_t_1 = o_t_1 * activation(c_t_1)  # actual hidden state

            return c_t, h_t, c_t_1, h_t_1

        [_, h_1, _, h_2], _ = theano.scan(
            recurrence,
            sequences=input,
            outputs_info=[T.alloc(self.c0, input.shape[1], hidden_dim),
                          T.alloc(self.h0, input.shape[1], hidden_dim),
                          T.alloc(self.c1, input.shape[1], hidden_dim),
                          T.alloc(self.h1, input.shape[1], hidden_dim)]
        )

        # since every hidden layer is connected to output
        self.y = T.concatenate([h_1[-1], h_2[-1]], axis=1)


class BiLstm(object):
    def __init__(self, input, vocab_size, emb_dim, hidden_dim, init='uniform', inner_init='orthonormal',
                 inner_activation=T.nnet.hard_sigmoid, activation=T.tanh,
                 params=None, merge_mode='concat'):
        input_f = input.dimshuffle(1, 0)
        input_b = input[::-1].dimshuffle(1, 0)
        if params is None:
            self.emb = theano.shared(value=get(identifier=init, shape=(vocab_size, emb_dim), scale=np.sqrt(3)),
                                     name='emb', borrow=True)
            # Forward LSTM
            # input gate
            self.Wf_i = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_i', borrow=True)
            self.Uf_i = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_i', borrow=True)
            self.bf_i = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                      name='bf_i', borrow=True)
            # forget gate
            self.Wf_f = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_f', borrow=True)
            self.Uf_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_f', borrow=True)
            self.bf_f = theano.shared(value=get(identifier='one', shape=(hidden_dim, )),
                                      name='bf_f', borrow=True)
            # memory
            self.Wf_c = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_c', borrow=True)
            self.Uf_c = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_c', borrow=True)
            self.bf_c = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                      name='bf_c', borrow=True)
            # output gate
            self.Wf_o = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_o', borrow=True)
            self.Uf_o = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_o', borrow=True)
            self.bf_o = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                      name='bf_o', borrow=True)

            # Backward LSTM
            # input gate
            self.Wb_i = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_i', borrow=True)
            self.Ub_i = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_i', borrow=True)
            self.bb_i = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                      name='bb_i', borrow=True)
            # forget gate
            self.Wb_f = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_f', borrow=True)
            self.Ub_f = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_f', borrow=True)
            self.bb_f = theano.shared(value=get(identifier='one', shape=(hidden_dim, )),
                                      name='bb_f', borrow=True)
            # memory
            self.Wb_c = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_c', borrow=True)
            self.Ub_c = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_c', borrow=True)
            self.bb_c = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                      name='bb_c', borrow=True)
            # output gate
            self.Wb_o = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_o', borrow=True)
            self.Ub_o = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_o', borrow=True)
            self.bb_o = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )),
                                      name='bb_o', borrow=True)

        else:
            self.emb, self.Wf_i, self.Uf_i, self.bf_i, self.Wf_f, self.Uf_f, self.bf_f, \
                self.Wf_c, self.Uf_c, self.bf_c, self.Wf_o, self.Uf_o, self.bf_o,\
                self.Wb_i, self.Ub_i, self.bb_i, self.Wb_f, self.Ub_f, self.bb_f, \
                self.Wb_c, self.Ub_c, self.bb_c, self.Wb_o, self.Ub_o, self.bb_o = params

        self.cf = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='cf', borrow=True)
        self.hf = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hf', borrow=True)
        self.cb = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='cb', borrow=True)
        self.hb = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hb', borrow=True)
        self.params = [self.emb,
                       self.Wf_i, self.Uf_i, self.bf_i,
                       self.Wf_f, self.Uf_f, self.bf_f,
                       self.Wf_c, self.Uf_c, self.bf_c,
                       self.Wf_o, self.Uf_o, self.bf_o,

                       self.Wb_i, self.Ub_i, self.bb_i,
                       self.Wb_f, self.Ub_f, self.bb_f,
                       self.Wb_c, self.Ub_c, self.bb_c,
                       self.Wb_o, self.Ub_o, self.bb_o]

        # forward lstm
        def recurrence_f(xf_t, cf_tm, hf_tm):
            xf_i = T.dot(self.emb[xf_t], self.Wf_i) + self.bf_i
            xf_f = T.dot(self.emb[xf_t], self.Wf_f) + self.bf_f
            xf_c = T.dot(self.emb[xf_t], self.Wf_c) + self.bf_c
            xf_o = T.dot(self.emb[xf_t], self.Wf_o) + self.bf_o

            if_t = inner_activation(xf_i + T.dot(hf_tm, self.Uf_i))
            ff_t = inner_activation(xf_f + T.dot(hf_tm, self.Uf_f))
            cf_t = ff_t * cf_tm + if_t * activation(xf_c + T.dot(hf_tm, self.Uf_c))  # internal memory
            of_t = inner_activation(xf_o + T.dot(hf_tm, self.Uf_o))
            hf_t = of_t * activation(cf_t)  # actual hidden state

            return cf_t, hf_t

        [_, self.h_f], _ = theano.scan(
            recurrence_f,
            sequences=input_f,
            outputs_info=[T.alloc(self.cf, input_f.shape[1], hidden_dim),
                          T.alloc(self.hf, input_f.shape[1], hidden_dim)]
        )

        # backward lstm
        def recurrence(xb_t, cb_tm, hb_tm):
            xb_i = T.dot(self.emb[xb_t], self.Wb_i) + self.bb_i
            xb_f = T.dot(self.emb[xb_t], self.Wb_f) + self.bb_f
            xb_c = T.dot(self.emb[xb_t], self.Wb_c) + self.bb_c
            xb_o = T.dot(self.emb[xb_t], self.Wb_o) + self.bb_o

            ib_t = inner_activation(xb_i + T.dot(hb_tm, self.Ub_i))
            fb_t = inner_activation(xb_f + T.dot(hb_tm, self.Ub_f))
            cb_t = fb_t * cb_tm + ib_t * activation(xb_c + T.dot(hb_tm, self.Ub_c))  # internal memory
            ob_t = inner_activation(xb_o + T.dot(hb_tm, self.Ub_o))
            hb_t = ob_t * activation(cb_t)  # actual hidden state

            return cb_t, hb_t

        [_, self.h_b], _ = theano.scan(
            recurrence,
            sequences=input_b,
            outputs_info=[T.alloc(self.cb, input_b.shape[1], hidden_dim),
                          T.alloc(self.hb, input_b.shape[1], hidden_dim)]
        )

        if merge_mode == 'sum':
            self.y = self.h_f[-1] + self.h_b[-1]
        elif merge_mode == 'multiply':
            self.y = self.h_f[-1] * self.h_b[-1]
        elif merge_mode == 'average':
            self.y = (self.h_f[-1] + self.h_b[-1]) / 2
        elif merge_mode == 'concat':
            self.y = T.concatenate([self.h_f[-1], self.h_b[-1]], axis=1)
        else:
            print('Supported "merge_mode" for forward + backward lstm are: "sum", "multiply", average" & "concat".')
            raise NotImplementedError
