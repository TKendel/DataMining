"""
#https://medium.com/@reddyyashu20/rnn-python-code-in-keras-and-pytorch-6ab842a85e15

from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN



# define the model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(None, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
# train the model
model.fit(x, y, batch_size=128, epochs=20)


# Define the model architecture
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(None, 1)))
model.add(Dense(units=1, activation='sigmoid'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
# Make predictions using the model
predictions = model.predict(x_new)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(None, 28)))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))
print(model.summary())


    #
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_validate, y_validate = x_test[:-10], y_test[:-10]
x_test, y_test = x_test[-10:], y_test[-10:]


    #Loss Function
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	