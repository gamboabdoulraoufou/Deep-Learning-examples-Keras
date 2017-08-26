
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
model.add(Dense(8, init='uniform', activation='relu')) # relu = rectified linear unit - f((x) = max(0,x)
model.add(Dense(1, init='uniform', activation='sigmoid')) # sigmoid = logistic function - f(x) = 1 / (1 - exp(-x))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
"""
Epoch 147/150
768/768 [==============================] - 0s - loss: 0.4544 - acc: 0.7852
Epoch 148/150
768/768 [==============================] - 0s - loss: 0.4560 - acc: 0.7865
Epoch 149/150
768/768 [==============================] - 0s - loss: 0.4685 - acc: 0.7826
Epoch 150/150
768/768 [==============================] - 0s - loss: 0.4548 - acc: 0.7904
"""

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
"""
acc: 79.56%
"""

# predict new value
predictions = model.predict(numpy.array([[6., 148., 72., 35., 0., 33.6, 0.627, 50.]]))
print(predictions)
"""
[[ 0.76625454]]
"""
