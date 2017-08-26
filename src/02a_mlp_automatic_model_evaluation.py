# MLP with automatic validation set
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("/Users/agambo/deep_learning/dpl_toolbox/data/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)

"""
Epoch 147/150
514/514 [==============================] - 0s - loss: 0.4928 - acc: 0.7685 - val_loss: 0.5036 - val_acc: 0.7598
Epoch 148/150
514/514 [==============================] - 0s - loss: 0.4915 - acc: 0.7646 - val_loss: 0.5157 - val_acc: 0.7520
Epoch 149/150
514/514 [==============================] - 0s - loss: 0.4780 - acc: 0.7802 - val_loss: 0.5105 - val_acc: 0.7559
Epoch 150/150
514/514 [==============================] - 0s - loss: 0.4820 - acc: 0.7821 - val_loss: 0.5559 - val_acc: 0.7362
"""
