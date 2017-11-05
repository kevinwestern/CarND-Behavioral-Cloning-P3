import csv
import numpy as np
import sklearn.utils
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D, Lambda, Reshape, Dropout
from keras.optimizers import Adam
from skimage import transform
from sklearn.model_selection import train_test_split

DATA_PATH = './data/'
CENTER = 0
LEFT = 1
RIGHT = 2
ANGLE = 3
LEFT_ANGLE_CORRECTION = 0.25
RIGHT_ANGLE_CORRECTION = -0.25

def get_lines(path):
    lines = []
    with open ('%sdriving_log.csv' % path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        return lines


def get_soft_sharp_split(data, threshold=0.3):
    soft_turns = []
    sharp_turns = []
    for line in data:
        if abs(float(line[3])) >= threshold:
            sharp_turns.append(line)
        else:
            soft_turns.append(line)
    return soft_turns, sharp_turns


def load_image(img_full_path):
    name = img_full_path.split('/')[-1]

    path = '%s/IMG/%s' % (DATA_PATH, name)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def random_translate(img, steering_angle, is_debug=False):
    random_x = np.random.randint(-50, 50)
    random_y = np.random.randint(-25, 25)
    steering_angle = steering_angle + random_x * 0.004
    tform = transform.SimilarityTransform(translation=(random_x, random_y))
    translated = transform.warp(img, tform)
    if is_debug:
        print('Random X: %s, Random Y: %s, Steering Angle Adjustment: %s' %(random_x, random_y, steering_angle))
    return translated, steering_angle


def random_brightness(img):
    img = np.array(img, dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    hsv[:,:,2] = hsv[:,:,2] * brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_training_image(line, camera=None):
    camera = np.random.randint(CENTER, RIGHT + 1) if camera is None else camera
    camera_img = load_image(line[camera])
    angle = float(line[ANGLE])

    steering_correction = 0.0

    if camera is LEFT:
        steering_correction = LEFT_ANGLE_CORRECTION
    elif camera is RIGHT:
        steering_correction = RIGHT_ANGLE_CORRECTION

    return camera_img, angle + steering_correction


def drop_random_low_angles(data, keep_probability=0.6, threshold=0.2):
    new_data = []
    for line in data:
        angle = abs(float(line[ANGLE]))
        if angle < threshold:
            if np.random.uniform() < keep_probability:
                new_data.append(line)
        else:
            new_data.append(line)
    return new_data


def make_validation_generator(data, batch_size=256):
    while True:
        num_samples = len(data)
        data = sklearn.utils.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img = load_image(batch_sample[CENTER])
                angle = float(batch_sample[ANGLE])
                images.append(img)
                angles.append(angle)
            x_images = np.array(images)
            y_labels = np.array(angles)
            yield x_images, y_labels


def make_train_generator(data, batch_size=256):
    while True:
        data = sklearn.utils.shuffle(data)
        num_samples = len(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:

                img, angle = get_training_image(batch_sample, camera=CENTER)
                
                images.append(img)
                angles.append(angle)
                
                #augmented_img = random_brightness(img)
                augmented_img, augmented_angle = random_translate(img, angle)

                # Maybe flip the image
                if np.random.randint(2):
                    augmented_img = np.fliplr(augmented_img)
                    augmented_angle *= -1

                images.append(augmented_img)
                angles.append(augmented_angle)

                left_camera, left_angle = get_training_image(batch_sample, camera=LEFT)
                right_camera, right_angle = get_training_image(batch_sample, camera=RIGHT)
                images.append(left_camera)
                images.append(right_camera)
                angles.append(left_angle)
                angles.append(right_angle)
                
            
            x_images = np.array(images)
            y_labels = np.array(angles)
            yield sklearn.utils.shuffle(x_images, y_labels)


def resize(img):
    import tensorflow as tf
    return tf.image.resize_images(img, (66, 200))


all_lines = get_lines(DATA_PATH)

train_samples, validation_samples = train_test_split(all_lines, test_size=0.2)
train_samples = drop_random_low_angles(train_samples, keep_probability=0.5, threshold=0.3)

soft, sharp = get_soft_sharp_split(train_samples, threshold=0.35)

# Add more sharp turns if needed
to_add = len(soft) - len(sharp)

# Add more sharp turns
for i in range(to_add):
    # Modulo so we can start from the beginning again
    sharp.append(sharp[i % len(sharp)])

train_samples = sklearn.utils.shuffle(sharp + soft)
train_num_samples = len(train_samples)

train_generator = make_train_generator(train_samples, batch_size=256)
validation_generator = make_validation_generator(validation_samples, batch_size=256)



model = Sequential()
# Shave 50px from the top and 20px from the bottom to be left with a 90x320x img
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Resize the image to match the nvidia sizes.
model.add(Lambda(resize))

# Normalize the image.
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), name='conv1', activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), name='conv2', activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), name='conv3', activation='relu'))
model.add(Convolution2D(64, 3, 3, name='conv4', activation='relu'))
model.add(Convolution2D(64, 3, 3, name='conv5', activation='relu'))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(1164, name='desn1164', activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, name='dense100'))
model.add(Dropout(0.5))
model.add(Dense(50, name='dense50'))
model.add(Dropout(0.5))
model.add(Dense(10, name='dense10'))
model.add(Dense(1, name='dense1'))

model.compile(loss='mse', optimizer=Adam(lr=0.001))
model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples) * 4,    # * 4 for center, left, and right image, augmented
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=5)

model.save('model.h5')
