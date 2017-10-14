""" ConvNet adult query classifier.

Heavily inspired by/cribbed from the Keras
imdb_cnn and pretrained_embeddings examples.
"""
from __future__ import print_function
from datetime import datetime
from time import time
import sys
# from keras import backend
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from utils_const_eval import *
from sklearn.metrics import classification_report
import ipdb

tick = time()

# get the datums
print('Loading/pre-processing data')
texts, labels = load_data()
sequences, word_index = tokenize_texts(texts)
data = pad_seqs(sequences)
# save indices for later indexing into texts
data, labels, indices = shuffle_data(data, labels)
x_train, y_train, x_val, y_val = train_val_split(data, labels)
print('#### %d secods elapsed ####' % (time() - tick,))


### MODEL DEFINITION
print('Defining model')
x = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
emb = Embedding(len(word_index) + 1,
                EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=True)(x)
conv_3 = Conv1D(FILTERS, 3, padding="valid",
                activation="relu", strides=1)(emb)
maxpool_3 = GlobalMaxPooling1D()(conv_3)
conv_4 = Conv1D(FILTERS, 4, padding="valid",
                activation="relu", strides=1)(emb)
maxpool_4 = GlobalMaxPooling1D()(conv_4)
conv_5 = Conv1D(FILTERS, 5, padding="valid",
                activation="relu", strides=1)(emb)
maxpool_5 = GlobalMaxPooling1D()(conv_5)
conv_6 = Conv1D(FILTERS, 6, padding="valid",
                activation="relu", strides=1)(emb)
maxpool_6 = GlobalMaxPooling1D()(conv_6)
# merged = concatenate([maxpool_3, maxpool_4, maxpool_5], axis=-1)
merged = concatenate([maxpool_3, maxpool_4, maxpool_5, maxpool_6], axis=-1)
hidden = Dense(HIDDEN_DIMS)(merged)
dropout = Dropout(P_DROPOUT)(hidden)
dropout = Activation("relu")(dropout)
out = Dense(1, activation="sigmoid")(dropout)

model = Model(inputs=x, outputs=out)
print('Compiling model')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print('##### %d seconds elapsed #####' % (time() - tick,))


print('Training char model with KERNEL_SIZES=%s' % str(KERNEL_SIZES))
callbacks = [EarlyStopping(patience=0, min_delta=0.001, verbose=1)]
data_size = 50000
print('On %d data points' % data_size)
hist = model.fit(x_train[:data_size], y_train[:data_size],
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 validation_data=(x_val, y_val),
                 callbacks=callbacks)
print('##### %d seconds elapsed #####' % (time() - tick,))

# serialize model to JSON and weights to HDF5
now = datetime.now().strftime("%Y%d_%H%M")
with open("models/model_%s.json" % now, "w") as json_file:
    json_file.write(model.to_json())
model.save_weights("models/model_%s.wts" % now)
print("Saved model to models/model_%s" % now)
print('##### %d seconds elapsed #####' % (time() - tick,))

print('Eval against val set')
tick = time()
# ipdb.set_trace()
print(classification_report(y_val,
                            np.round(model.predict(x_val)).astype('int32'),
                            digits=3))
print('predict() on %d items in %d seconds' % (x_val.shape[0], time() - tick))
texts = np.asarray(texts)
eval_against_lr(model,
                texts[indices][-2*x_val.shape[0]:-x_val.shape[0]],
                labels[-2*x_val.shape[0]:-x_val.shape[0]])
