# Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("/Users/agambo/deep_learning/dpl_toolbox/data/pima-indians-diabetes.csv", delimiter=",")


"""
1- Train model on small set of trainning examples
"""
# split into input (X) and output (Y) variables
X = dataset[:100,0:8]
Y = dataset[:100,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.layers[2].get_weights()
"""
[array([[-0.04884608],
       [ 0.03666496],
       [-0.0230276 ],
       [-0.02516407],
       [-0.01400621],
       [-0.03158818],
       [-0.0187674 ],
       [-0.02188858]], dtype=float32), array([ 0.], dtype=float32)]
"""

# checkpoint
filepath="/Users/agambo/deep_learning/dpl_toolbox/data/checkpoint/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=1, batch_size=1, callbacks=callbacks_list, verbose=0)
model.layers[2].get_weights()

# Evaluation
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


"""
acc: 62.00%

[array([[-0.05522409],
       [ 0.03547617],
       [-0.05009582],
       [-0.03267258],
       [-0.02088056],
       [-0.03757844],
       [-0.0187674 ],
       [-0.02188858]], dtype=float32), array([-0.04001966], dtype=float32)]
"""

"""
3- Train model on the rest of trainning examples by using stored weight
"""
# split into input (X) and output (Y) variables
X = dataset[100:,0:8]
Y = dataset[100:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# load weights
model.load_weights("/Users/agambo/deep_learning/dpl_toolbox/data/checkpoint/weights.best.hdf5")

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.layers[2].get_weights()

# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=1, batch_size=1, callbacks=callbacks_list, verbose=1)
model.layers[2].get_weights()
"""
[array([[-0.54559332],
       [ 0.01558436],
       [-0.20072663],
       [-0.16023931],
       [-0.35260549],
       [-0.33703417],
       [-0.04892033],
       [-0.05566839]], dtype=float32), array([ 0.06816341], dtype=float32)]
"""
# Evaluation
X = dataset[:,0:8]
Y = dataset[:,8]
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
"""
acc: 65.10%
"""
