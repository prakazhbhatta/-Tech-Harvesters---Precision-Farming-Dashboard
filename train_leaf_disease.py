import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'valid'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper to build a simple CNN
def build_model(num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data generators
all_labels = os.listdir(DATA_DIR)
potato_labels = [d for d in all_labels if d.lower().startswith('potato')]
tomato_labels = [d for d in all_labels if d.lower().startswith('tomato')]

# Train for each crop
def train_and_save(labels, model_name):
    print(f"\nTraining model for: {model_name}")
    # Create a subdirectory with only the relevant classes
    temp_dir = f'_temp_{model_name}'
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    for label in labels:
        src = os.path.join(DATA_DIR, label)
        dst = os.path.join(temp_dir, label)
        import shutil
        shutil.copytree(src, dst)
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(temp_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
    val_gen = datagen.flow_from_directory(temp_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')
    model = build_model(len(labels))
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    model.save(os.path.join(MODEL_DIR, f'{model_name}_leaf_model.h5'))
    print(f"Model saved: {MODEL_DIR}/{model_name}_leaf_model.h5")
    # Clean up temp dir
    shutil.rmtree(temp_dir)

if potato_labels:
    train_and_save(potato_labels, 'potato')
if tomato_labels:
    train_and_save(tomato_labels, 'tomato') 