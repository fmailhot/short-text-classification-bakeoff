import sys
import os
import errno
import shutil
import requests
from zipfile import Zipfile


def fetch_data(path="./data"):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    try:
        #TODO(fmailhot): check if local file exists before doing this
        requests.get("http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip")
    except Exception as e:
        sys.exit("Failed fetching data file: %s" % str(e))
    with Zipfile("Sentiment-Analysis-Dataset.zip") as myzip:
        myzip.extractall("data/")
    os.remove("Sentiment-Analysis-Dataset.zip")
    return


def load_data():
    print('Train/val data loading')
    texts = []
    labels = []
    for f in glob("data/*"):
        if "clean" in f:
            label = 0
        else:
            label = 1
        for line in open(f):
            texts.append(line)
            labels.append(label)
    print('Found %d texts' % (len(texts),))
    return texts, labels


def tokenize_texts(texts):
    print('Tokenizing')
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %d unique tokens' % (len(word_index),))
    return sequences, word_index


def pad_seqs(sequences):
    print('Padding, encoding, train/dev split')
    data = pad_sequences(sequences,
                         maxlen=MAX_SEQUENCE_LENGTH,
                         padding='post',
                         truncating='post')
    return data


def train_val_split(data, labels):
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-(nb_validation_samples*2)]
    x_val = data[-(nb_validation_samples*2):-nb_validation_samples]
    y_train = labels[:-(nb_validation_samples*2)]
    y_val = labels[-(nb_validation_samples*2):-nb_validation_samples]
    print('Shape of training set:', x_train.shape)
    print('Shape of validation set:', x_val.shape)
    return x_train, y_train, x_val, y_val
