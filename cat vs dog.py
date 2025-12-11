pip install -q Kaggle 
from google.colab import files
files.upload()
Saving kaggle.json to kaggle.json
{'kaggle.json': b'{"username":"abhilashkr27","key":"66c1851a5616f94857e3b3e78989af1c"}'}
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d tongpython/cat-and-dog -p /content

import zipfile

zip_ref = zipfile.ZipFile("/content/cat-and-dog.zip", 'r')
zip_ref.extractall("/content")
zip_ref.close()
!ls /content
cat-and-dog.zip  kaggle.json  sample_data  test_set  training_set
!ls /content/cat-and-dog
ls: cannot access '/content/cat-and-dog': No such file or directory!ls /content/training_set
raining_set
# Move cats and dogs folders out of the inner training_set folder
!mv /content/training_set/training_set/* /content/training_set/

# Delete the now-empty extra folder
!rm -r /content/training_set/training_se
!mv /content/test_set/test_set/* /content/test_set/
!rm -r /content/test_set/test_set
mv: cannot stat '/content/test_set/test_set/*': No such file or directory
rm: cannot remove '/content/test_set/test_set': No such file or directory
!ls /content/training_set
!ls /content/test_set
cats  dogs
cats  dogs
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import layers
import matplotlib.pyplot as plt
train_ds = keras.utils.image_dataset_from_directory(
    directory="/content/training_set",
    labels='inferred',
    label_mode='int',
    image_size=(256,256),
    batch_size=32
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory="/content/test_set",
    labels='inferred',
    label_mode='int',
    image_size=(256,256),
    batch_size=32
)
Found 8005 files belonging to 2 classes.
Found 2023 files belonging to 2 classes.
# Normalize the data
def process(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])
model = Sequential([
    data_augmentation,

    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=20, validation_data=validation_ds)
[ ]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


[ ]
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='red', label='Training Loss')
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Save model
model.save("cats_vs_dogs_augmented.keras")
print("✅ Improved Model Saved Successfully!")

# Optional: download
from google.colab import files
files.download("cats_vs_dogs_augmented.keras")
from keras.models import load_model
model = load_model("/content/cats_vs_dogs_final.keras")
print("✅ Model loaded successfully!")
ValueError                                Traceback (most recent call last)
/tmp/ipython-input-1051098230.py in <cell line: 0>()
      1 from keras.models import load_model
----> 2 model = load_model("/content/cats_vs_dogs_final.keras")
      3 print("✅ Model loaded successfully!")

/usr/local/lib/python3.12/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible .keras "

ValueError: File not found: filepath=/content/cats_vs_dogs_final.keras. Please ensure the file is an accessible .keras zip file.
!ls /content
cat-and-dog.zip		      kaggle.json  test_set
cats_vs_dogs_augmented.keras  sample_data  training_set
from google.colab import files
uploaded = files.upload()

Saving cat.jpeg to cat.jpeg
Saving dog.jpg to dog.jpg

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test both images (cat + dog)
for img_path in ["/content/cat.jpeg", "/content/dog.jpg"]:
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(img_path.split('/')[-1])
    plt.axis("off")
    plt.show()

    img_resized = cv2.resize(img, (256,256))
    img_input = img_resized.reshape((1,256,256,3)) / 255.0
    output = model.predict(img_input)

    label = "🐶 Dog" if output > 0.5 else "🐱 Cat"
    print(f"{img_path.split('/')[-1]} → {label}\n")