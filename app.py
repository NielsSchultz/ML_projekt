import string
import matplotlib.pyplot as plt
import os
import subprocess
import re
import shutil
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
import datetime


#NAME = "Reviews-pos-neg-cnn-{}".format(int(time.time()))
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#downloader og extracter mappen med reviews
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

#opretter mapper og undermapper til data
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

#udskriver et eksempel på et review


#fjerner unsup mappen, da den ikke bliver brugt
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#batch size = hvor mange af gangen
batch_size = 32
seed = 42

#Note: When using the validation_split and subset arguments, make sure to either specify a random seed, or to pass shuffle=False, 
#so that the validation and training splits have no overlap.
#opretter trænings datasættet ud fra mappen train og angiver subsettet bruges til træning
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)
#opretter validerings datasættet og angiver subsettet skal bruges til validering
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

#opretter test datasættet
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)

#standardiserer indhold(fjerner html,lowercase,whitespaces)
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


max_features = 10000
sequence_length = 250
#opretter textvectorization layer https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
#og sætter output mode til int og bruger den standardiseringsmodel som er definereret oven over
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

#adapt() method on a dataset. When this layer is adapted, it will analyze the dataset, determine the frequency of individual string values, 
# and create a 'vocabulary' from them. This vocabulary can have unlimited size or be capped, depending on the configuration options for this layer; 
# if there are more unique values in the input than the maximum vocabulary size, the most frequent terms will be used to create the vocabulary.
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


#preprocessing step, apply the TextVectorization layer created earlier to the train, validation, and test dataset.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#configuring datasets
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

#configure the model to use an optimizer and a loss function. https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
#Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#train model ved at smide datasættet ind in fit metoden
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=[tensorboard_callback],
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

#udskriver loss og accuracy
print("Loss: ", loss)
print("Accuracy: ", accuracy)

#model.fit() returns a History object that contains a dictionary with everything that happened during training:
history_dict = history.history
history_dict.keys()

#plotting loss over epochs
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#plotting accuracy over epochs 
#Stop training at peak accuracy https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])
#Computes the cross-entropy loss between true labels and predicted labels.
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile


#gemmer den trænede model til en temp mappe med versionsnavn
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)