
## Deep Learning Demo
In this post, I show an example of using deep learning techniques to identify objects on images. The example assume that you are familiar with the theory of the neural networks and Python.

I will use Convolutional Neural Networks (CNN or Conv Net) algorithm which are very similar to ordinary Neural Networks and CIFAR-10 dataset (60000 32x32 colour images in 10 classes, with 6000 images per class).

In `src` folder, you can find more examples for:
- Regression and supervised learning using Multilayer Perceptron (MLP)
- Image classification with advanced features (image augmentation, ...) using Convolutional Neural Networks (CNN or Conv Net)
- Time series or text generation using Recurrent Neural Networks (RNN ans specialy LSTM)

All input datasets are ind `data` folder or Keras package (so you need internet access for Keras dataset like CIFAR-10)


### Overview
- Requirements
- Load Python modules and datasets
- Prepare data
- Model architecture
- Model Training
- Model evaluation

### 1- Requirements
- Python 2.7
- Numpy
- Matplotlib
- keras
- TensorFlow or Theano
- Internet

### 2- Load Python modules and datasets


```python
# import modules
import os
import numpy as np

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
```

    Using Theano backend.



```python
# change some package default options
%matplotlib inline 
pyplot.switch_backend('agg')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K.set_image_dim_ordering('th')
```


```python
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
```


```python
# Load data
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
```


```python
# Show Examples from Each Class
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  train_features.shape
num_classes = len(np.unique(train_labels))

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = pyplot.figure(figsize=(10,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(class_names[i])
    pyplot.imshow(im)
pyplot.show()
```


```python
fig
```




![png](img/output_9_0.png)




```python
pyplot.show()
```

### 3- Prepare data


```python
# normalize inputs from 0-255 to 0.0-1.0 to optimize algorithm convergence
X_train = train_features.astype('float32')
X_train = X_train / 255.0
y_train = train_labels
X_test = test_features.astype('float32')
X_test = X_test / 255.0
y_test = test_labels
```


```python
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

### 4- Define model architecture and compile


```python
# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
```

    -c:3: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(3, 32, 32...)`
    -c:5: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), padding="same", activation="relu")`
    -c:7: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3), padding="same", activation="relu")`
    -c:9: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(64, (3, 3), padding="same", activation="relu")`
    -c:11: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(128, (3, 3), padding="same", activation="relu")`
    -c:13: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(128, (3, 3), padding="same", activation="relu")`
    -c:17: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(1024, activation="relu", kernel_constraint=<keras.con...)`
    -c:19: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(512, activation="relu", kernel_constraint=<keras.con...)`



```python
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 32, 16, 16)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 64, 16, 16)        18496     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64, 16, 16)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 64, 16, 16)        36928     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 64, 8, 8)          0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 128, 8, 8)         73856     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 128, 8, 8)         0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 128, 8, 8)         147584    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 128, 4, 4)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2048)              0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              2098176   
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               524800    
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 2,915,114
    Trainable params: 2,915,114
    Non-trainable params: 0
    _________________________________________________________________
    None


### 5- Train modele


```python
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
```

    /usr/local/anaconda2/lib/python2.7/site-packages/keras/models.py:844: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      warnings.warn('The `nb_epoch` argument in `fit` '


    Train on 50000 samples, validate on 10000 samples
    Epoch 1/25
    50000/50000 [==============================] - 919s - loss: 1.9505 - acc: 0.2793 - val_loss: 1.7162 - val_acc: 0.3888
    Epoch 2/25
    50000/50000 [==============================] - 932s - loss: 1.5321 - acc: 0.4406 - val_loss: 1.3784 - val_acc: 0.5012
    Epoch 3/25
    50000/50000 [==============================] - 908s - loss: 1.3440 - acc: 0.5116 - val_loss: 1.3328 - val_acc: 0.5138
    Epoch 4/25
    50000/50000 [==============================] - 904s - loss: 1.2117 - acc: 0.5645 - val_loss: 1.1271 - val_acc: 0.5956
    Epoch 5/25
    50000/50000 [==============================] - 941s - loss: 1.1001 - acc: 0.6081 - val_loss: 1.0158 - val_acc: 0.6345
    Epoch 6/25
    50000/50000 [==============================] - 932s - loss: 1.0097 - acc: 0.6398 - val_loss: 0.9641 - val_acc: 0.6574
    Epoch 7/25
    50000/50000 [==============================] - 924s - loss: 0.9342 - acc: 0.6692 - val_loss: 0.8748 - val_acc: 0.6892
    Epoch 8/25
    50000/50000 [==============================] - 931s - loss: 0.8637 - acc: 0.6921 - val_loss: 0.8274 - val_acc: 0.7126
    Epoch 9/25
    50000/50000 [==============================] - 892s - loss: 0.8112 - acc: 0.7127 - val_loss: 0.8064 - val_acc: 0.7128
    Epoch 10/25
    50000/50000 [==============================] - 940s - loss: 0.7638 - acc: 0.7315 - val_loss: 0.7567 - val_acc: 0.7327
    Epoch 11/25
    50000/50000 [==============================] - 897s - loss: 0.7201 - acc: 0.7429 - val_loss: 0.7377 - val_acc: 0.7448
    Epoch 12/25
    50000/50000 [==============================] - 907s - loss: 0.6843 - acc: 0.7586 - val_loss: 0.7082 - val_acc: 0.7526
    Epoch 13/25
    50000/50000 [==============================] - 909s - loss: 0.6503 - acc: 0.7699 - val_loss: 0.6843 - val_acc: 0.7615
    Epoch 14/25
    50000/50000 [==============================] - 926s - loss: 0.6251 - acc: 0.7788 - val_loss: 0.6705 - val_acc: 0.7673
    Epoch 15/25
    50000/50000 [==============================] - 904s - loss: 0.5964 - acc: 0.7890 - val_loss: 0.6657 - val_acc: 0.7681
    Epoch 16/25
    50000/50000 [==============================] - 933s - loss: 0.5735 - acc: 0.7967 - val_loss: 0.6679 - val_acc: 0.7712
    Epoch 17/25
    50000/50000 [==============================] - 972s - loss: 0.5514 - acc: 0.8042 - val_loss: 0.6388 - val_acc: 0.7773
    Epoch 18/25
    50000/50000 [==============================] - 909s - loss: 0.5343 - acc: 0.8101 - val_loss: 0.6440 - val_acc: 0.7754
    Epoch 19/25
    50000/50000 [==============================] - 943s - loss: 0.5152 - acc: 0.8169 - val_loss: 0.6209 - val_acc: 0.7859
    Epoch 20/25
    50000/50000 [==============================] - 917s - loss: 0.4965 - acc: 0.8229 - val_loss: 0.6209 - val_acc: 0.7872
    Epoch 21/25
    50000/50000 [==============================] - 897s - loss: 0.4780 - acc: 0.8295 - val_loss: 0.6263 - val_acc: 0.7897
    Epoch 22/25
    50000/50000 [==============================] - 892s - loss: 0.4630 - acc: 0.8346 - val_loss: 0.6139 - val_acc: 0.7949
    Epoch 23/25
    50000/50000 [==============================] - 907s - loss: 0.4510 - acc: 0.8407 - val_loss: 0.6128 - val_acc: 0.7953
    Epoch 24/25
    50000/50000 [==============================] - 923s - loss: 0.4368 - acc: 0.8459 - val_loss: 0.6064 - val_acc: 0.7970
    Epoch 25/25
    50000/50000 [==============================] - 905s - loss: 0.4217 - acc: 0.8470 - val_loss: 0.5984 - val_acc: 0.7992





    <keras.callbacks.History at 0x7f9e37bd3d50>



### 6- Evaluate model


```python
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    Accuracy: 79.92%



```python

```
