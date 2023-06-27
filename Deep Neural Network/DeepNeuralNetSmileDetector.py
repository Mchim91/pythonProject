from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

directory = '../training_set/'

pixel_intensities = []
# one-hot encoding happy (1,0) and sad (0, 1)
labels = []

for filename in os.listdir(directory):
    image = Image.open(directory+filename).convert('1')
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] == 'happy':
        labels.append([1, 0])
    elif filename[0:3] == 'sad':
        labels.append([0, 1])

pixel_intensities = np.array(pixel_intensities)
labels = np.array(labels)

# apply min-max normalization (here just /255)
pixel_intensities = pixel_intensities / 255.0
print(pixel_intensities)

# create the model (deep neural networks)
model = Sequential()
model.add(Dense(1024, input_dim=1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(pixel_intensities, labels, epochs=1000, batch_size=20, verbose=2)
