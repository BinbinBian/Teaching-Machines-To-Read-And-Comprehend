import random

__author__ = 'uyaseen'


# shuffle the lists of s, s_mask, q, q_mask & a
def shuffle(s, s_mask, q, q_mask, a):
    data = zip(s, s_mask, q, q_mask, a)
    random.shuffle(data)
    s, s_mask, q, q_mask, a = zip(*data)
    return list(s), list(s_mask), list(q), list(q_mask), list(a)


def get_padded_text(text, pad_idx, max_length):
    # left padding
    return [pad_idx] * (max_length - len(text)) + text, \
           [0] * (max_length - len(text)) + [1] * len(text)


# overly simplistic/naive tokenization
def tokenize(text):
    return text.split(' ')
