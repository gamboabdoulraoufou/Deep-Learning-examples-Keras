# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("/Users/agambo/deep_learning/dpl_toolbox/data/pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed) # create model

# model parameter
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)
"""
Epoch 147/150
514/514 [==============================] - 0s - loss: 0.4936 - acc: 0.7685 - val_loss: 0.5426 - val_acc: 0.7283
Epoch 148/150
514/514 [==============================] - 0s - loss: 0.4957 - acc: 0.7685 - val_loss: 0.5430 - val_acc: 0.7362
Epoch 149/150
514/514 [==============================] - 0s - loss: 0.4953 - acc: 0.7685 - val_loss: 0.5403 - val_acc: 0.7323
Epoch 150/150
514/514 [==============================] - 0s - loss: 0.4941 - acc: 0.7743 - val_loss: 0.5452 - val_acc: 0.7323
"""
