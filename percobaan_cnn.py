import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import urllib.request
import zipfile
import os

# Download dataset (Rock Paper Scissors) if not exists
url = 'https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip'
local_zip = 'rockpaperscissors.zip'
if not os.path.exists(local_zip):
    print("Mengunduh dataset...")
    urllib.request.urlretrieve(url, local_zip)
    print("Mengekstrak dataset...")
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall('.')

base_dir = 'rockpaperscissors/rps-cv-images'

# Persiapan Data dengan ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode='wrap',
    validation_split=0.4 # Sesuai standar untuk membagi validation data
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    class_mode='categorical',
    subset='validation'
)

# 5. Membuat arsitektur model CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Menampilkan summary model
model.summary()

# 6 & 7. Kompilasi model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 8. Proses pelatihan model (Model Fitting)
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Evaluasi model
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')

# 9. Prediksi hasil model
predictions = model.predict(validation_generator)
print(predictions) # Output berupa probabilitas prediksi tiap kelas
