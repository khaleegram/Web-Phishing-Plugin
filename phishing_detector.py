import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, GRU, Dense, Dropout, BatchNormalization, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Optimize TensorFlow for CPU
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"

# Enable XLA Optimization
tf.config.optimizer.set_jit(True)

# Enable Mixed Precision (if supported)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Load Dataset
data = pd.read_csv("Phishing_new 2023.csv")
np.random.seed(0)
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

# Split Features & Target
X = data.drop(columns=["label"], errors="ignore")
Y = data["label"]
X = X.select_dtypes(include=[np.number])

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Feature Selection
top_k_features = 48
selector = SelectKBest(f_classif, k=top_k_features)
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

# Reshape Data for CNN-BiGRU
X_train_reshaped = X_train_selected.reshape((X_train_selected.shape[0], X_train_selected.shape[1], 1))
X_test_reshaped = X_test_selected.reshape((X_test_selected.shape[0], X_test_selected.shape[1], 1))

# Define Hybrid CNN-BiGRU Model
def create_hybrid_cnn_bigru_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # CNN Branch
    cnn_branch = Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')(input_layer)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
    cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
    cnn_branch = Flatten(name="cnn_features")(cnn_branch)
    
    # BiGRU Branch
    bigru_branch = Bidirectional(GRU(128, return_sequences=True))(input_layer)
    bigru_branch = Bidirectional(GRU(64, return_sequences=False), name="bigru_features")(bigru_branch)
    bigru_branch = Dropout(0.5)(bigru_branch)
    
    # Concatenate CNN & BiGRU Features
    merged = Concatenate()([cnn_branch, bigru_branch])
    dense_layer = Dense(128, activation='relu')(merged)
    dense_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize Model
hybrid_model = create_hybrid_cnn_bigru_model((X_train_reshaped.shape[1], 1))

# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Optimize Batch Size
batch_size = 32  # Experiment with 32 or 64

steps_per_epoch = min(len(X_train_reshaped) // batch_size, 200)

# Train Model
history = hybrid_model.fit(
    X_train_reshaped, Y_train,
    epochs=5,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[early_stopping],
    steps_per_epoch=steps_per_epoch,
    verbose=1
)


# Save Model
hybrid_model.save("hybrid_cnn_bigru_model.h5")
print("Model saved successfully!")

# Make Predictions
Y_pred_prob = hybrid_model.predict(X_test_reshaped)
Y_pred = (Y_pred_prob > 0.5).astype(int)

# Evaluate Model Performance
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Print Results
print("\nHybrid CNN-BiGRU Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
plt.show()

# Load Model for Future Use
loaded_model = load_model("hybrid_cnn_bigru_model.h5")
print("Model loaded successfully!")