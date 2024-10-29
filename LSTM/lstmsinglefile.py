# classification_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Concatenate, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the dataset
df = pd.read_csv("dataset.csv")
df.index = pd.to_datetime(df["id"], errors="ignore")

# Define the columns for training
cols = [
    "press",
    "humid",
    "temp",
    "ws100",
    "verts100",
    "wdir100",
    "cis6",
    "cis7",
    "wdisp100",
    "vertdisp100"
]

df_for_training = df[cols].astype(float)

# Feature Engineering: Create lagged features
for lag in range(1, 3):  # Lag of 1 to 2 time steps (adjusted for computational efficiency)
    lagged_df = df_for_training.shift(lag).add_suffix(f'_lag_{lag}')
    df_for_training = pd.concat([df_for_training, lagged_df], axis=1)

# Drop rows with NaN values caused by shifting
df_for_training.dropna(inplace=True)

# Scaling with MinMaxScaler
scaler = MinMaxScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.save')

# Define sequence and prediction lengths
sequence_length = 36  # Last 6 hours (36 values)
prediction_length = 6  # Next 1 hour (6 values)

X, y = [], []

# Prepare sequences for input and output
for i in range(len(df_for_training_scaled) - sequence_length - prediction_length + 1):
    X.append(df_for_training_scaled[i: i + sequence_length])
    y.append(df_for_training_scaled[i + sequence_length: i + sequence_length + prediction_length, :len(cols)])  # Predicting original cols

X = np.array(X)
y = np.array(y)

print(f"Shape of X: {X.shape}")  # (samples, 36, features)
print(f"Shape of y: {y.shape}")  # (samples, 6, features)

# Create binary target variable for 'ws100'
ws100_index = cols.index('ws100')
y_ws100 = y[:, :, ws100_index]
y_binary = (y_ws100 < 6.0).astype(int)  # Binary classification target

# Reshape classification targets to match the model output shape
y_binary = y_binary.reshape(-1, prediction_length, 1)

# Manual train-test split
train_size = int(len(X) * 0.8)
X_train = X[:train_size]
y_train_regression = y[:train_size]
y_train_classification = y_binary[:train_size]

X_test = X[train_size:]
y_test_regression = y[train_size:]
y_test_classification = y_binary[train_size:]

# Build the model
units = 128  # Number of LSTM units
learning_rate = 1e-3

# Encoder
encoder_inputs = Input(shape=(sequence_length, X.shape[2]))
encoder_outputs, state_h, state_c = LSTM(
    units=units,
    return_sequences=True,
    return_state=True
)(encoder_inputs)

# Decoder
decoder_inputs = RepeatVector(prediction_length)(state_h)
decoder_outputs, _, _ = LSTM(
    units=units,
    return_sequences=True,
    return_state=True
)(decoder_inputs, initial_state=[state_h, state_c])

# Attention
attention_outputs = Attention()([decoder_outputs, encoder_outputs])

# Concatenate decoder outputs with attention outputs
decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_outputs])

# Regression output
regression_output = TimeDistributed(Dense(len(cols), activation='linear'), name='regression_output')(decoder_concat_input)

# Classification output (ensure units=1)
classification_output = TimeDistributed(Dense(1, activation='sigmoid'), name='classification_output')(decoder_concat_input)

# Define the model
model = Model(inputs=encoder_inputs, outputs=[regression_output, classification_output])

# Compile the model with multiple loss functions
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss={
        'regression_output': 'mse',
        'classification_output': 'binary_crossentropy'
    },
    loss_weights={
        'regression_output': 1.0,
        'classification_output': 1.0
    },
    metrics={
        'regression_output': ['mse'],
        'classification_output': ['accuracy']
    }
)

model.summary()

# Training the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    X_train,
    {
        'regression_output': y_train_regression,
        'classification_output': y_train_classification
    },
    validation_data=(X_test, {
        'regression_output': y_test_regression,
        'classification_output': y_test_classification
    }),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save('classification_model.h5')

# Evaluate the model
# Make predictions
predictions = model.predict(X_test)
y_pred_regression = predictions[0]
y_pred_classification = predictions[1]

# Reshape predictions and true values for classification
y_pred_classification_flat = y_pred_classification.reshape(-1)
y_test_classification_flat = y_test_classification.reshape(-1)

# Threshold the classification predictions at 0.5
y_pred_classification_binary = (y_pred_classification_flat > 0.5).astype(int)

# Compute classification metrics
print("Classification Report for 'ws100' below 6 m/s:")
print(classification_report(y_test_classification_flat, y_pred_classification_binary))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_classification_flat, y_pred_classification_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Optionally, plot the confusion matrix
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Above 6 m/s', 'Below 6 m/s'], yticklabels=['Above 6 m/s', 'Below 6 m/s'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")
plt.show()

# Evaluate regression performance on 'ws100'
# Inverse transform to original scale

# Reshape y_pred_regression and y_test_regression to 2D arrays for inverse scaling
y_pred_regression_reshaped = y_pred_regression.reshape(-1, len(cols))
y_test_regression_reshaped = y_test_regression.reshape(-1, len(cols))

# Prepare full arrays for inverse scaling
num_features = df_for_training.shape[1]  # Total number of features after lagging
padding = np.zeros((y_pred_regression_reshaped.shape[0], num_features - len(cols)))
y_pred_full = np.hstack([y_pred_regression_reshaped, padding])
y_test_full = np.hstack([y_test_regression_reshaped, padding])

# Inverse scaling
y_pred_orig = scaler.inverse_transform(y_pred_full)[:, :len(cols)]
y_test_orig = scaler.inverse_transform(y_test_full)[:, :len(cols)]

# Extract 'ws100' predictions and true values
y_pred_ws100 = y_pred_orig[:, ws100_index]
y_test_ws100 = y_test_orig[:, ws100_index]

# Compute RMSE for 'ws100'
rmse_ws100 = np.sqrt(np.mean((y_test_ws100 - y_pred_ws100) ** 2))
print(f"RMSE for 'ws100': {rmse_ws100:.4f} m/s")

# Plot actual vs predicted 'ws100' values
plt.figure(figsize=(10,6))
plt.plot(y_test_ws100, label='Actual ws100')
plt.plot(y_pred_ws100, label='Predicted ws100')
plt.legend()
plt.title('Actual vs Predicted ws100')
plt.xlabel('Sample')
plt.ylabel('ws100 (m/s)')
plt.show()

# Evaluate regression performance specifically when 'ws100' is below 6 m/s
below_threshold_indices = np.where(y_test_ws100 < 6.0)
rmse_below_threshold = np.sqrt(np.mean((y_test_ws100[below_threshold_indices] - y_pred_ws100[below_threshold_indices]) ** 2))
print(f"RMSE for 'ws100' below 6 m/s: {rmse_below_threshold:.4f} m/s")
