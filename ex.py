import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==============================
# 1. Dataset Paths
# ==============================

train_dir = "dataset/train"
val_dir = "dataset/val"

img_size = (224, 224)
batch_size = 32

# ==============================
# 2. Load Dataset
# ==============================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_dataset.class_names
print("Classes:", class_names)

# ==============================
# 3. Prefetch for Performance
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ==============================
# 4. Load Pretrained Model
# ==============================

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False   # Freeze base layers

# ==============================
# 5. Build Classification Model
# ==============================

model = keras.Sequential([
    
    base_model,
    
    layers.GlobalAveragePooling2D(),
    
    layers.BatchNormalization(),
    
    layers.Dense(128, activation='relu'),
    
    layers.Dropout(0.5),
    
    layers.Dense(len(class_names), activation='softmax')
])

# ==============================
# 6. Compile Model
# ==============================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 7. Train Model
# ==============================

epochs = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# ==============================
# 8. Save Model
# ==============================

model.save("document_classifier_model.h5")

print("Model saved successfully!")

# ==============================
# 9. Plot Accuracy
# ==============================

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()