from utilities.datareader import load_data

from model.deep_reader import train_deep
from model.attentive_reader import train_attentive

__author__ = 'uyaseen'


def train_data(reader, model, task):
    path = 'data/' + task + '/'
    vocab, data = load_data(path=path, task=task)
    n_epochs = 10000  # Note: Set this accordingly ...
    optimizer = 'rmsprop'
    emb_dim = 100
    hidden_dim = 256
    batch_size = 32
    if reader == 'deep':
        train_deep(data, vocab, task=task, emb_dim=emb_dim, hidden_dim=hidden_dim, b_path=path,
                   rec_model=model, optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size,
                   use_existing_model=True)
    elif reader == 'attentive':
        attn_dim = 100
        train_attentive(data, vocab, task=task, emb_dim=emb_dim, hidden_dim=hidden_dim, attn_dim=attn_dim,
                        b_path=path, rec_model=model, optimizer=optimizer, n_epochs=n_epochs,
                        batch_size=batch_size, use_existing_model=True)

    return 0

if __name__ == '__main__':
    train_data(reader='deep', model='deep-gru', task='cnn')
    print('... done')
