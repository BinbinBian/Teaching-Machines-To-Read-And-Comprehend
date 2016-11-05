import os
import cPickle as pkl
from text_utils import tokenize, get_padded_text, shuffle

__author__ = 'uyaseen'


# declare few tokens
unknown_token = 'UNKNOWN_TOKEN'
pad_token = 'PADDING'

max_story_length = -1
max_query_length = -1


def read_raw_data(in_file, max_example=None, relabeling=True):
    """
        source: https://github.com/danqi/rc-cnn-dailymail/blob/master/code/utils.py
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    global max_story_length
    global max_query_length
    stories = []
    questions = []
    answers = []
    num_examples = 0
    f = open(in_file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        story = f.readline().strip().lower()

        if relabeling:
            q_words = tokenize(question)
            s_words = tokenize(story)
            assert answer in s_words

            entity_dict = {}
            entity_id = 0
            for word in s_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            s_words = [entity_dict[w] if w in entity_dict else w for w in s_words]
            answer = entity_dict[answer]

            question = ' '.join(q_words)
            story = ' '.join(s_words)

            story_len = len(tokenize(story)) + 2
            query_len = len(tokenize(question)) + 2
            if story_len > max_story_length:
                max_story_length = story_len
            if query_len > max_query_length:
                max_query_length = query_len

        questions.append(question)
        answers.append(answer)
        stories.append(story)
        num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    return [stories, questions, answers]


def build_vocab(sentences):
    vocab = set()
    for sent in sentences:
        for word in tokenize(sent):
            vocab.add(word)

    vocab.add(unknown_token)
    print('vocabulary size: %i' % len(vocab))
    return vocab


def vectorize(data, vocab):
    story, question, answer = data
    global max_story_length
    global max_query_length
    data_s = []
    data_s_mask = []
    data_q = []
    data_q_mask = []
    data_a = []
    words_to_ix = {wd: i+1 for i, wd in enumerate(vocab)}
    words_to_ix.update({pad_token: 0})
    for s, q, a in zip(story, question, answer):
        s = [words_to_ix[wd] if wd in vocab else words_to_ix[unknown_token]
             for wd in tokenize(s)]
        q = [words_to_ix[wd] if wd in vocab else words_to_ix[unknown_token]
             for wd in tokenize(q)]
        # Pad to cater for max sentence length
        s, s_mask = get_padded_text(s, words_to_ix[pad_token], max_story_length)
        q, q_mask = get_padded_text(q, words_to_ix[pad_token], max_query_length)
        a = words_to_ix[a] if a in vocab else words_to_ix[unknown_token]
        data_s.append(s)
        data_s_mask.append(s_mask)
        data_q.append(q)
        data_q_mask.append(q_mask)
        data_a.append(a)

    # shuffle the data
    data_s, data_s_mask, data_q, data_q_mask, data_a = shuffle(data_s, data_s_mask,
                                                               data_q, data_q_mask,
                                                               data_a)

    return [data_s, data_s_mask, data_q, data_q_mask, data_a]


def pickle_data(path, w_path, task):
    train = read_raw_data(path + 'train.txt')
    print('# of examples in Train set: %i' % len(train[0]))
    valid = read_raw_data(path + 'dev.txt')
    print('# of examples in Validation set: %i' % len(valid[0]))
    test = read_raw_data(path + 'test.txt')
    print('# of examples in Test set: %i' % len(test[0]))
    voc = build_vocab(train[0] + train[1])
    train = vectorize(train, voc)
    valid = vectorize(valid, voc)
    test = vectorize(test, voc)
    data = [train, valid, test]

    words_to_ix = {wd: i+1 for i, wd in enumerate(voc)}
    ix_to_words = {i+1: wd for i, wd in enumerate(voc)}
    # enforce 'PADDING' index to be 0
    words_to_ix.update({pad_token: 0})
    ix_to_words.update({0: pad_token})
    voc.add(pad_token)
    vocab = [voc, words_to_ix, ix_to_words]

    print '... creating persistence storage'
    curr_dir = os.getcwd()
    vocab_path = w_path + task + '_dict.pkl'
    f = open(vocab_path, 'wb')
    pkl.dump(vocab, f, -1)
    f.close()

    data_path = w_path + task + '_data.pkl'
    f = open(data_path, 'wb')
    pkl.dump(data, f, -1)
    f.close()

    os.chdir(curr_dir)
    print '"%s" & "%s" created' % (vocab_path, data_path)


def load_data(path, task):
        print '... loading pickled data'

        curr_dir = os.getcwd()
        vocab_path = path + task + '_dict.pkl'
        f = open(vocab_path, 'rb')
        vocab = pkl.load(f)
        f.close()

        data_path = path + task + '_data.pkl'
        f = open(data_path, 'rb')
        data = pkl.load(f)
        f.close()

        os.chdir(curr_dir)
        return vocab, data
