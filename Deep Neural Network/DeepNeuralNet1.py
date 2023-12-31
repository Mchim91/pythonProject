import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Why XOR? Because it's a non-linearly separable problem
# XOR problem training samples
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

# XOR problem target values accordingly
target_data = np.array([[0], [1], [1], [0]], "float32")

# we can define the neural network layers in a sequential manner
model = Sequential()
# first parameter is output dimension
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for the weights: SGD, Ada-grad, ADAM ...
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# epoch is an iteration over the entire dataset
# verbose 0 is silent 1 and 2 are showing results
model.fit(training_data, target_data, epochs=500, verbose=2)

# of course we can make predictions with the trained neural network
print(model.predict(training_data).round())
