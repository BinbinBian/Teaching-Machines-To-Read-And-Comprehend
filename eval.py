from utilities.datareader import load_data
from utilities.model_utils import evaluate_attentive, evaluate_deep

__author__ = 'uyaseen'


def eval_model(reader, model, task, emb_dim=100, hidden_dim=256, attn_dim=100):
    path = 'data/' + task + '/'
    vocab, data = load_data(path=path, task=task)
    if reader == 'deep':
        print('** Deep Reader **')
        evaluate_deep(m_path=path + '/models/' + task + '_' + model + '_best_model.pkl',
                      test_data=data[2], vocabulary=vocab, emb_dim=emb_dim,
                      hidden_dim=hidden_dim, model=model)
    elif reader == 'attentive':
        print('** Attentive Reader **')
        evaluate_attentive(m_path=path + '/models/' + task + '_' + model + '_best_model.pkl',
                           test_data=data[2], vocabulary=vocab, emb_dim=emb_dim,
                           hidden_dim=hidden_dim, attn_dim=attn_dim, model=model)
    else:
        print('Only `deep` & `attentive` readers are supported.')
        return TypeError


if __name__ == '__main__':
    task = 'cnn'
    eval_model(reader='deep', model='deep-gru', task=task)
    print('... done')
