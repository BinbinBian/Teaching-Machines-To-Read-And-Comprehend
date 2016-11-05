import theano
import theano.tensor as T

from utilities.initializations import get

__author__ = 'uyaseen'


class LogisticRegression(object):

    def __init__(self, input, input_dim, output_dim, params):
        if params is None:
            self.W = theano.shared(
                value=get(identifier='uniform', shape=(input_dim, output_dim)),
                name='w', borrow=True)
            self.b = theano.shared(
                value=get(identifier='zero', shape=(output_dim,)),
                name='b', borrow=True)

        else:
            self.W, self.b = params

        self.params = [self.W, self.b]

        self.p_y_given_x = T.clip(T.nnet.softmax(T.dot(input, self.W) + self.b),
                                  0.0001, 0.9999)  # need 'clipping' to avoid nan in nll

        self.pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.neq(self.pred, y)


class MLPAttn(object):
    def __init__(self, story, question, hidden_dim, output_dim,
                 attn_dim, params, activation=T.tanh):
        story = story.dimshuffle(1, 0, 2)
        if params is None:
            self.W_att_story = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, attn_dim)),
                name='W_att_story',
                borrow=True
            )
            self.W_att_question = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, attn_dim)),
                name='W_att_question',
                borrow=True
            )
            # weight matrix for 'm_t' (see original paper: page 5)
            self.W_m = theano.shared(
                value=get(identifier='uniform', shape=(attn_dim, )),
                name='W_m',
                borrow=True
            )
            self.W_rg = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='W_rg',
                borrow=True
            )
            self.W_ug = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='W_ug',
                borrow=True
            )
            self.b = theano.shared(
                value=get(identifier='zero', shape=(output_dim,)),
                name='b',
                borrow=True
            )
        else:
            self.W_att_story, self.W_att_question, self.W_m, self.W_rg, self.W_ug, self.b = params

        self.params = [self.W_att_story, self.W_att_question, self.W_m, self.W_rg, self.W_ug, self.b]
        
        # applying attention i.e weighted sum of story & question
        def step(token_t):
            m_t = activation(T.dot(token_t, self.W_att_story) +
                             T.dot(question, self.W_att_question))
            # attention at time-step t (will be a scalar value)
            s_t = T.dot(m_t, self.W_m)  # is 'W_m' even needed here ?
            return s_t

        s, _ = theano.scan(
            step,
            sequences=story,
            outputs_info=None
        )
        s = s.dimshuffle(1, 0)
        # normalized attention
        s_norm = T.nnet.softmax(s)

        # embedding of 'story'

        def compute_batch_sum(story_, norm_):
            return story_.T * norm_

        r_t, _ = theano.scan(
            compute_batch_sum,
            sequences=[story, s_norm.dimshuffle(1, 0)],
            outputs_info=None
        )

        r = T.sum(r_t.dimshuffle(2, 0, 1), axis=1)

        # given 'r' & 'u' ; compute the final 'g' (where 'u' = encoding of question)
        self.g = T.nnet.softmax(activation(T.dot(r, self.W_rg) +
                                           T.dot(question, self.W_ug) + self.b))

        self.pred = T.argmax(self.g, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.g)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.neq(self.pred, y)
