

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================
# CONFIG
# ============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "./dataset"
MODEL_PATH = "disaster_model_best.h5"

# ============================
# DATA GENERATORS
# ============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = len(train_gen.class_indices)
CLASS_MAP = {v: k for k, v in train_gen.class_indices.items()}

# ============================
# CLASS WEIGHTS (IMPORTANT)
# ============================
from sklearn.utils.class_weight import compute_class_weight

labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

# ============================
# MODEL (TRANSFER LEARNING)
# ============================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# CALLBACKS
# ============================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# ============================
# TRAIN
# ============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ============================
# FINE TUNING (UNFREEZE)
# ============================
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# ============================
# SAVE FINAL MODEL
# ============================
model.save("disaster_model_final.h5")

# ============================
# PLOT HISTORY
# ============================
def plot_history(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title("Loss")

    plt.show()

plot_history(history)

# ============================
# EVALUATION
# ============================
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=list(CLASS_MAP.values())))

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_MAP.values(),
            yticklabels=CLASS_MAP.values())
plt.title("Confusion Matrix")
plt.show()

# ============================
# PREDICTION FUNCTION
# ============================
def predict_disaster(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    confidence = preds[0][idx]

    return CLASS_MAP[idx], confidence

# ============================
# GRAD-CAM (EXPLAINABILITY)
# ============================
def grad_cam(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.output, model.layers[-4].output]
    )

    with tf.GradientTape() as tape:
        preds, conv_outputs = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.show()

# ============================
# TEST
# ============================
if __name__ == "__main__":

    test_img = "./dataset/test/sample_fire.jpg"

    pred, conf = predict_disaster(test_img)

    print(f"Prediction: {pred} ({conf:.2f})")

    grad_cam(test_img)