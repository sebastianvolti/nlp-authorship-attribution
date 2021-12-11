from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, SpatialDropout1D, Conv1D
from keras.layers import GlobalMaxPooling1D, Bidirectional, Embedding
from keras import backend as K
import numpy as np
import tensorflow as tf
from constants import MODEL_TYPES

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def getMaxSize(dataset):
  max_size = 0
  for line in dataset:
    line_size = len(line)
    if (line_size > max_size):
      max_size = line_size

  return max_size

class Model():

  # We need the val_dataset in the constructor to find the max size
  def __init__(self, model_type, train_dataset, val_dataset, test_dataset, neurons, dropout):
    self.train_dataset = train_dataset["Paragraph"]
    self.train_labels = np.array(train_dataset["Author"])
    self.val_dataset = val_dataset["Paragraph"]
    self.val_labels = np.array(val_dataset["Author"])
    self.test_dataset = None
    self.test_labels = None
    self.max_size =  max(getMaxSize(train_dataset["Paragraph"]), getMaxSize(val_dataset["Paragraph"]))
    self.model = None
    self.neurons=neurons
    self.dropout=dropout
    self.train_padded_docs = None
    self.type = model_type
    self.initModel()

  def chooseModel(self, vocab_size, embedding_matrix):
    model = Sequential()
    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=self.max_size, trainable=False)
    model.add(e)

    if (self.type == MODEL_TYPES["SIMPLE"]):
      model.add(Flatten())

    elif (self.type == MODEL_TYPES["LSTM1"]):
      model.add(LSTM(self.neurons, dropout=self.dropout))

    elif (self.type == MODEL_TYPES["LSTM2"]):
      model.add(SpatialDropout1D(0.25))
      model.add(LSTM(self.neurons, dropout=self.dropout, recurrent_dropout=self.dropout))
      model.add(Dropout(0.2))

    elif (self.type == MODEL_TYPES["BIDIRECTIONAL"]):
      model.add(Bidirectional(LSTM(self.neurons,dropout=self.dropout)))

    else:
      model.add(Conv1D(self.neurons, 5, activation='relu'))
      model.add(GlobalMaxPooling1D())
    
    # Add last layer
    model.add(Dense(5, activation='softmax'))
    
    # Compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
    #model.compile(loss=['mae', 'sparse_categorical_crossentropy'], optimizer='adam')
    #model.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

  def train(self, epochs, batchs):
    if (self.type == MODEL_TYPES["SIMPLE"]):
      self.model.fit(self.train_padded_docs, self.train_labels, epochs=epochs, verbose=1)
    else:
       self.model.fit(self.train_padded_docs, self.train_labels, epochs=epochs, batch_size=batchs, verbose=1)

  def eval(self):
    if(self.test_dataset):
      print("test files..")

    else:
      print("val files..")
      result = []
      t = Tokenizer()
      t.fit_on_texts(self.val_dataset)
      encoded_docs = t.texts_to_sequences(self.val_dataset)
      padded_docs = pad_sequences(encoded_docs, maxlen=self.max_size, padding='post')
      #loss, accuracy, f1_score, precision, recall = self.model.evaluate(padded_docs, self.val_labels, verbose=1)
      result = self.model.evaluate(padded_docs, self.val_labels, verbose=1)
      #predictions = self.model.predict(padded_docs, verbose=1)

      #print('Accuracy: %f' % (accuracy*100))
      #print('Precision: %f' % (precision*100))
      #print('Recall: %f' % (recall*100))
      #print('F1: %f' % (f1_score*100))
      #return (accuracy*100,f1_score*100,recall*100,precision*100)
    
  
  def initModel(self):
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(self.train_dataset)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(self.train_dataset)
    self.train_padded_docs = np.array(pad_sequences(encoded_docs, maxlen=self.max_size, padding='post'))
    # load the whole embedding into memory
    embeddings_index = dict()
    final = 'parte_3_redes/resources/fasttext.es.300.txt'
    f = open('parte_3_redes/resources/fasttext.es.300.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector 
    
    model = self.chooseModel(vocab_size, embedding_matrix)
    # summarize the model
    print(model.summary())
    self.model = model
