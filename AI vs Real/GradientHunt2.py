import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ReLU,AveragePooling2D
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from keras.regularizers import l2

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    
    for image in tqdm(images):
        try:
            img = load_img(image, target_size=(224,224))
            img = np.array(img)
            features.append(img)
        except Exception as e:
            print(f"Skipping file {image}: {e}")
    features = np.array(features)
    features = features.reshape(features.shape[0], 224,224, 3) 
    return features

TRAIN_DIR = "/kaggle/input/dataset/Data/Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

x_train = extract_features(train['image'])/255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

import numpy as np

indices = np.arange(len(x_train))
np.random.shuffle(indices)

x_train = x_train[indices]
y_train = y_train[indices]


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224,224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(2, activation='softmax'))


adam = keras.optimizers.Adam(learning_rate=0.001,clipnorm=0.8)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=25,batch_size=16,verbose=2)
import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img


TEST_DIR = "/kaggle/input/task2data/induction-task-2025/Test_Images"
test_image_paths = [os.path.join(TEST_DIR, img) for img in os.listdir(TEST_DIR)]


def preprocess_test_images(image_paths):
    features = []
    ids = []
    for image_path in image_paths:
        try: 
            img = load_img(image_path, target_size=(224,224))
            
            features.append(img)
            ids.append(os.path.basename(image_path))
        except Exception as e:
            print(f"Skipping file {image_path}: {e}")
    return np.array(features), ids

test_features, test_ids = preprocess_test_images(test_image_paths)

predictions = model.predict(test_features/255.0)
predicted_labels = np.argmax(predictions, axis=1)

class_names = le.classes_  
mapped_labels = [class_names[label] for label in predicted_labels]


submission = pd.DataFrame({
    "Id": [img.split('.')[0] for img in test_ids],
    "Label": mapped_labels
})

submission.to_csv("submission2.csv", index=False)
print("Submission file saved as 'submission.csv'")
