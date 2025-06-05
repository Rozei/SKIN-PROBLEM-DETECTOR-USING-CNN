import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
import tensorflow as tf

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# === 1. Load Labels CSV ===
csv_path = r'C:\Users\imroz\Music\skin problem detect\data\_classes.csv'
df = pd.read_csv(csv_path)

# === 2. Append full image path to 'filename' column ===
image_dir = r'C:\Users\imroz\Music\data\train'
df['filename'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))

# === 3. Split the dataset (80% train, 20% validation) ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# === 4. Image Preprocessing ===
image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col=list(df.columns[1:]),  # All label columns
    target_size=image_size,
    batch_size=batch_size,
    class_mode='raw'  # for multi-label classification
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col=list(df.columns[1:]),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='raw'
)

# === 5. Build CNN Model ===
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(df.columns[1:]), activation='sigmoid')  # Multi-label output
])

# === 6. Compile Model ===
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === 7. Train Model ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# === 8. Save Model ===
model.save(r'C:\Users\imroz\Music\skin_problem_model.h5')
