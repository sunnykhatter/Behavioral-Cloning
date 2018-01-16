import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation,Cropping2D,Convolution2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



#Helper methods for generatore functions
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
            	# Center Image
                name = "./IMG/"+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle)
                angles.append(center_angle*-1.0)
                # # Left Image
                name = "./IMG/"+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                images.append(left_image)
                images.append(cv2.flip(left_image,1))
                angles.append(center_angle + correction)
                angles.append((center_angle + correction) *-1.0)
                # # Right Image
                name = "./IMG/"+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                images.append(right_image)
                images.append(cv2.flip(right_image,1))
                angles.append(center_angle - correction)
                angles.append((center_angle - correction) *-1.0)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print(X_train.shape, y_train.shape)
            yield (X_train, y_train)


#Importing Data into Memory
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

samples = lines[1:]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


print("Number of Samples: " + str(len(train_samples)))
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Create the CNN model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# #Print a summary of the model
print(model.summary())


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Compile the model, using the training and set and save the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=25)
model.save('model1.h5')

