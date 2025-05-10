import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ✅ Step 1: Set up paths
BASE_DIR = r"C:\Users\sarth\Downloads\lathaproject"
METADATA_PATH = os.path.join(BASE_DIR, "Data_Entry_2017.csv")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

# ✅ Step 2: Load metadata
metadata = pd.read_csv(METADATA_PATH)
print("✅ Metadata loaded!")

# ✅ Step 3: Collect all image paths
image_paths, labels = [], []
for folder in os.listdir(BASE_DIR):
    if folder.startswith("images_"):
        folder_path = os.path.join(BASE_DIR, folder, "images")
        if not os.path.exists(folder_path):
            continue
        for image_file in os.listdir(folder_path):
            if image_file.endswith(".png"):
                image_path = os.path.join(folder_path, image_file)
                image_paths.append(image_path)
                label_row = metadata.loc[metadata["Image Index"] == image_file, "Finding Labels"].values
                labels.append(label_row[0].split("|") if len(label_row) > 0 else ["No Finding"])

print(f"✅ Total images found: {len(image_paths)}")

# ✅ Step 4: Convert to DataFrame
df = pd.DataFrame({"Image Path": image_paths, "Finding Labels": labels})

# ✅ Step 5: Select 6 most important diseases
IMPORTANT_DISEASES = ["Pneumonia", "Pneumothorax", "Cardiomegaly", "Pleural_thickening", "Fibrosis", "No Finding"]
for disease in IMPORTANT_DISEASES:
    df[disease] = df["Finding Labels"].apply(lambda x: 1 if disease in x else 0)

df = df.drop_duplicates(subset=["Image Path"])
df = df[df[IMPORTANT_DISEASES].sum(axis=1) > 0]  # Keep images with at least one label

print(f"✅ Diseases retained for training: {IMPORTANT_DISEASES}")

# ✅ Step 6: Split dataset
df["Stratify_Label"] = df[IMPORTANT_DISEASES].idxmax(axis=1)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Stratify_Label"], random_state=42)

print(f"✅ Train: {len(train_df)}, Validation: {len(val_df)}")

# ✅ Step 7: Data Generators (with augmentation)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col="Image Path", y_col=IMPORTANT_DISEASES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="raw"
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col="Image Path", y_col=IMPORTANT_DISEASES,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="raw"
)

print("✅ Data generators ready!")

# ✅ Step 8: Load EfficientNetB4 & fine-tune last 150 layers
base_model = EfficientNetB4(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False  # Initially freeze the model

def unfreeze_model(model, trainable_layers=150):
    for layer in model.layers[-trainable_layers:]:
        layer.trainable = True

unfreeze_model(base_model, trainable_layers=150)

# ✅ Step 9: Build custom model architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.4),  # Slightly higher dropout for better generalization
    layers.Dense(len(IMPORTANT_DISEASES), activation="sigmoid")  # Multi-label classification
])

# ✅ Step 10: Define Optimizer with CosineDecay
lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=1000, alpha=0.1)

model.compile(
    optimizer=AdamW(learning_rate=lr_schedule),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),  # For multi-label classification
    metrics=["accuracy"]
)

# ✅ Step 11: Train model with only EarlyStopping
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
]

history = model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=callbacks)
print("✅ Model training complete!")

# ✅ Step 12: Evaluate Model
loss, accuracy = model.evaluate(val_generator)
print(f"✅ Final Accuracy: {accuracy:.4f}")

# ✅ Step 13: Plot Training History
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.grid(True)
plt.show()

import tensorflow.keras.backend as K
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# ✅ Step 14: Grad-CAM Utilities
def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    return img_array

def generate_gradcam_heatmap(model, img_array, class_idx, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# ✅ Step 15: Apply Grad-CAM to One Image
sample_image_path = val_df["Image Path"].values[0]  # Try any image from validation set
img_array = load_image(sample_image_path)
preds = model.predict(img_array)[0]

# ✅ Display Grad-CAM for all predicted positive classes
for i, prob in enumerate(preds):
    if prob > 0.5:
        heatmap = generate_gradcam_heatmap(model, img_array, i, last_conv_layer_name="top_conv")
        superimposed_img = overlay_heatmap(sample_image_path, heatmap)

        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Grad-CAM: {IMPORTANT_DISEASES[i]} (Confidence: {prob:.2f})")
        plt.axis("off")
        plt.show()
