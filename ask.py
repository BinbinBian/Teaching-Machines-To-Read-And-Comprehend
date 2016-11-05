from utilities.datareader import load_data
from utilities.model_utils import answer_deep, answer_attentive

__author__ = 'uyaseen'


def ask_model(reader, model, task, emb_dim=100, hidden_dim=256, attn_dim=100):
    path = 'data/' + task + '/'
    vocab, data = load_data(path=path, task=task)
    if reader == 'deep':
        print('** Deep Reader **')
        answer_deep(m_path=path + '/models/' + task + '_' + model + '_best_model.pkl',
                    vocabulary=vocab, emb_dim=emb_dim, hidden_dim=hidden_dim,
                    model=model)
    elif reader == 'attentive':
        print('** Attentive Reader **')
        answer_attentive(m_path=path + '/models/' + task + '_' + model + '_best_model.pkl',
                         vocabulary=vocab, emb_dim=emb_dim, hidden_dim=hidden_dim,
                         attn_dim=attn_dim, model=model)
    else:
        print('Only `deep` & `attentive` readers are supported.')
        return TypeError


if __name__ == '__main__':
    task = 'cnn'
    ask_model(reader='deep', model='deep-gru', task=task)
    print('... done')
