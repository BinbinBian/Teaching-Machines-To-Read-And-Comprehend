# Teaching-Machines-to-Read-and-Comprehend

Theano implementation of **Deep LSTM Reader** & **Attentive Reader** from Google DeepMind's paper [Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340) - Hermann et al. (2015).
![Models](/doc/models.png?raw=true)

## Dependencies
* Python 2.7
* [Numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [scikit-learn](http://scikit-learn.org/stable/) (for computing F1 score)


## Datasets
* [rc-data](https://github.com/deepmind/rc-data)

I am using processed RC datasets from [this](https://github.com/danqi/rc-cnn-dailymail#datasets) repository. 
The original datasets can be downloaded from [https://github.com/deepmind/rc-data](https://github.com/deepmind/rc-data) or [http://cs.nyu.edu/~kcho/DMQA/](http://cs.nyu.edu/~kcho/DMQA/).
Processed ones are just simply concatenation of all data instances and keeping document, question and answer only.

## Usage

**Note:** story & question are alias for document & query respectively.

### Generating Data
```bash
python data.py
```

### Training
`train.py` provides an easy interface to train deep/attentive reader, `model/*_reader.py`  contains the actual code for model definition and training. Please note that call to `train_*(..)` using `use_existing_model=True` will replace the current existing best model with the new best model, so save your intermediate models accordingly.

### Evaluation

`eval.py` provide interface to compute various performance params (accuracy, f1-score) for trained models.
```bash
python eval.py
```

Use `ask.py` to let the model infer from your stories and questions.
```bash
python ask.py
```

### Acknowledgement
This code uses portion of Data reading interface written by [Danqi Chen](https://github.com/danqi).
