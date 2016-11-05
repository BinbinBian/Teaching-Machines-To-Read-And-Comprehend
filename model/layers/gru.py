import theano
import theano.tensor as T

import numpy as np

from utilities.initializations import get

__author__ = 'uyaseen'


# Stack GRU with skip-connections.
class DeepGru(object):
    def __init__(self, input, vocab_size, emb_dim, hidden_dim, n_layers=2, init='uniform',
                 inner_init='orthonormal', inner_activation=T.nnet.hard_sigmoid,
                 activation=T.tanh, params=None):
        input = input.dimshuffle(1, 0)
        assert(n_layers == 2)  # can only stack one layer
        if params is None:
            self.emb = theano.shared(value=get(identifier=init, shape=(vocab_size, emb_dim), scale=np.sqrt(3)),
                                     name='emb', borrow=True)
            # Layer 1
            # update gate
            self.W_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_z', borrow=True)
            self.U_z = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_z', borrow=True)
            self.b_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_z', borrow=True)
            # reset gate
            self.W_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_r', borrow=True)
            self.U_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_r', borrow=True)
            self.b_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_r', borrow=True)
            # hidden state
            self.W_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='W_h', borrow=True)
            self.U_h = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                     name='U_h', borrow=True)
            self.b_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                     name='b_h', borrow=True)
            # Layer 2
            # update gate
            self.W_z_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_z_1', borrow=True)
            self.U_z_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_z_1', borrow=True)
            self.b_z_1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                       name='b_z_1', borrow=True)
            # reset gate
            self.W_r_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_r_1', borrow=True)
            self.U_r_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_r_1', borrow=True)
            self.b_r_1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                       name='b_r_1', borrow=True)
            # hidden state
            self.W_h_1 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim)),
                                       name='W_h_1', borrow=True)
            self.U_h_1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                       name='U_h_1', borrow=True)
            self.b_h_1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                       name='b_h_1', borrow=True)
            # Skip-connections from input to layer 2
            self.s_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='s_z', borrow=True)
            self.s_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='s_r', borrow=True)
            self.s_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                     name='s_h', borrow=True)
        else:
            self.emb, self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r, self.W_h, self.U_h, \
                self.b_h, self.W_z_1, self.U_z_1, self.b_z_1, self.W_r_1, self.U_r_1, self.b_r_1, \
                self.W_h_1, self.U_h_1, self.b_h_1, self.s_z, self.s_r, self.s_h = params

        self.h0 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h0', borrow=True)
        self.h1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='h1', borrow=True)
        self.params = [self.emb,
                       self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W_h, self.U_h, self.b_h,
                       self.W_z_1, self.U_z_1, self.b_z_1,
                       self.W_r_1, self.U_r_1, self.b_r_1,
                       self.W_h_1, self.U_h_1, self.b_h_1,
                       self.s_z, self.s_r, self.s_h]

        def recurrence(x_t, h_t1_prev, h_t2_prev):
            # Layer 1
            x_z_1 = T.dot(self.emb[x_t], self.W_z) + self.b_z
            x_r_1 = T.dot(self.emb[x_t], self.W_r) + self.b_r
            x_h_1 = T.dot(self.emb[x_t], self.W_h) + self.b_h

            z_t_1 = inner_activation(x_z_1 + T.dot(h_t1_prev, self.U_z))
            r_t_1 = inner_activation(x_r_1 + T.dot(h_t1_prev, self.U_r))
            hh_t_1 = activation(x_h_1 + T.dot(r_t_1 * h_t1_prev, self.U_h))
            h_t_1 = (T.ones_like(z_t_1) - z_t_1) * hh_t_1 + z_t_1 * h_t1_prev

            # Layer 2
            # 's_*' represents skip connections from previous layer
            x_z_2 = T.dot(h_t_1, self.W_z_1) + T.dot(self.emb[x_t], self.s_z) + self.b_z_1
            x_r_2 = T.dot(h_t_1, self.W_r_1) + T.dot(self.emb[x_t], self.s_r) + self.b_r_1
            x_h_2 = T.dot(h_t_1, self.W_h_1) + T.dot(self.emb[x_t], self.s_h) + self.b_h_1

            z_t_2 = inner_activation(x_z_2 + T.dot(h_t2_prev, self.U_z_1))
            r_t_2 = inner_activation(x_r_2 + T.dot(h_t2_prev, self.U_r_1))
            hh_t_2 = activation(x_h_2 + T.dot(r_t_2 * h_t2_prev, self.U_h_1))
            h_t_2 = (T.ones_like(z_t_2) - z_t_2) * hh_t_2 + z_t_2 * h_t2_prev

            return h_t_1, h_t_2

        [h_1, h_2], _ = theano.scan(
            recurrence,
            sequences=input,
            outputs_info=[T.alloc(self.h0, input.shape[1], hidden_dim),
                          T.alloc(self.h1, input.shape[1], hidden_dim)]
        )

        # since every hidden layer is connected to output
        self.y = T.concatenate([h_1[-1], h_2[-1]], axis=1)


class BiGru(object):
    def __init__(self, input, vocab_size, emb_dim, hidden_dim, init='uniform', inner_init='orthonormal',
                 inner_activation=T.nnet.hard_sigmoid, activation=T.tanh,
                 params=None, merge_mode='concat'):
        input_f = input.dimshuffle(1, 0)
        input_b = input[::-1].dimshuffle(1, 0)
        if params is None:
            self.emb = theano.shared(value=get(identifier=init, shape=(vocab_size, emb_dim), scale=np.sqrt(3)),
                                     name='emb', borrow=True)
            # Forward GRU
            # update gate
            self.Wf_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_z', borrow=True)
            self.Uf_z = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_z', borrow=True)
            self.bf_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bf_z', borrow=True)
            # reset gate
            self.Wf_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_r', borrow=True)
            self.Uf_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_r', borrow=True)
            self.bf_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bf_r', borrow=True)
            # hidden state
            self.Wf_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wf_h', borrow=True)
            self.Uf_h = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Uf_h', borrow=True)
            self.bf_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bf_h', borrow=True)

            # Backward GRU
            # update gate
            self.Wb_z = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_z', borrow=True)
            self.Ub_z = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_z', borrow=True)
            self.bb_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bb_z', borrow=True)
            # reset gate
            self.Wb_r = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_r', borrow=True)
            self.Ub_r = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_r', borrow=True)
            self.bb_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bb_r', borrow=True)
            # hidden state
            self.Wb_h = theano.shared(value=get(identifier=init, shape=(emb_dim, hidden_dim)),
                                      name='Wb_h', borrow=True)
            self.Ub_h = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                      name='Ub_h', borrow=True)
            self.bb_h = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                      name='bb_h', borrow=True)

        else:
            self.emb, self.Wf_z, self.Uf_z, self.bf_z, self.Wf_r, self.Uf_r, self.bf_r, self.Wf_h, self.Uf_h, \
                self.bf_h, self.Wb_z, self.Ub_z, self.bb_z, self.Wb_r, self.Ub_r, self.bb_r, self.Wb_h, \
                self.Ub_h, self.bb_h = params

        self.hf = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hf', borrow=True)
        self.hb = theano.shared(value=get(identifier='zero', shape=(hidden_dim, )), name='hb', borrow=True)
        self.params = [self.emb,
                       self.Wf_z, self.Uf_z, self.bf_z,
                       self.Wf_r, self.Uf_r, self.bf_r,
                       self.Wf_h, self.Uf_h, self.bf_h,

                       self.Wb_z, self.Ub_z, self.bb_z,
                       self.Wb_r, self.Ub_r, self.bb_r,
                       self.Wb_h, self.Ub_h, self.bb_h]

        # forward gru
        def recurrence_f(xf_t, hf_tm):
            xf_z = T.dot(self.emb[xf_t], self.Wf_z) + self.bf_z
            xf_r = T.dot(self.emb[xf_t], self.Wf_r) + self.bf_r
            xf_h = T.dot(self.emb[xf_t], self.Wf_h) + self.bf_h

            zf_t = inner_activation(xf_z + T.dot(hf_tm, self.Uf_z))
            rf_t = inner_activation(xf_r + T.dot(hf_tm, self.Uf_r))
            hhf_t = activation(xf_h + T.dot(rf_t * hf_tm, self.Uf_h))
            hf_t = (T.ones_like(zf_t) - zf_t) * hhf_t + zf_t * hf_tm

            return hf_t

        self.h_f, _ = theano.scan(
            recurrence_f,
            sequences=input_f,
            outputs_info=T.alloc(self.hf, input_f.shape[1], hidden_dim)
        )

        # backward gru
        def recurrence_b(xb_t, hb_tm):
            xb_z = T.dot(self.emb[xb_t], self.Wb_z) + self.bb_z
            xb_r = T.dot(self.emb[xb_t], self.Wb_r) + self.bb_r
            xb_h = T.dot(self.emb[xb_t], self.Wb_h) + self.bb_h

            zb_t = inner_activation(xb_z + T.dot(hb_tm, self.Ub_z))
            rb_t = inner_activation(xb_r + T.dot(hb_tm, self.Ub_r))
            hhb_t = activation(xb_h + T.dot(rb_t * hb_tm, self.Ub_h))
            hb_t = (T.ones_like(zb_t) - zb_t) * hhb_t + zb_t * hb_tm

            return hb_t

        self.h_b, _ = theano.scan(
            recurrence_b,
            sequences=input_b,
            outputs_info=T.alloc(self.hb, input_b.shape[1], hidden_dim)
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
            print('Supported "merge_mode" for forward + backward gru are: "sum", "multiply", average" & "concat".')
            raise NotImplementedError
