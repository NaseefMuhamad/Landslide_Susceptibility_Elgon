"""
Script 03: Model Training and Evaluation
=========================================

This script handles:
- Building the CNN-LSTM hybrid model architecture
- Training the model on prepared datasets
- Model evaluation and validation
- Saving the trained model

Outputs to models/:
- cnn_lstm_model.h5 (trained model)
"""

# TODO: Implement CNN-LSTM model training and evaluation logic

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# --- 1. CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'final_dataset.csv')
MODEL_PATH = os.path.join(ROOT_DIR, '..', 'models', 'cnn_lstm_model.h5')
RESULTS_PATH = os.path.join(ROOT_DIR, '..', 'results', 'model_performance.txt')

# Define features based on Script 02 outputs
STATIC_FEATURES = ['elevation', 'slope', 'aspect', 'twi']
SOIL_FEATURE = ['soil_class'] # Categorical feature
TEMPORAL_FEATURES = [col for col in pd.read_csv(DATA_FILE, nrows=1).columns if 'rainfall' in col]

TIMESTEPS = len(TEMPORAL_FEATURES)
N_STATIC = len(STATIC_FEATURES) + 1 # +1 for one-hot encoded soil data

# --- 2. DATA LOADING AND PREPARATION ---

def load_and_prepare_data():
    """Loads the dataset, handles encoding, scaling, and array reshaping."""
    print("⏳ Loading and preparing data...")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ ERROR: Dataset not found at {DATA_FILE}. Run Script 02 first.")
        return None, None, None, None, None, None

    df = pd.read_csv(DATA_FILE)
    
    # Drop auxiliary columns
    df.drop(columns=['latitude', 'longitude'], inplace=True)
    df.dropna(inplace=True) # Final clean-up for any remaining NaNs

    # --- Feature Separation ---
    X_static_raw = df[STATIC_FEATURES]
    X_soil_raw = df[SOIL_FEATURE]
    X_temporal_raw = df[TEMPORAL_FEATURES]
    Y = df['landslide'].values

    # 1. Scale/Normalize Static Features
    scaler_static = StandardScaler()
    X_static_scaled = scaler_static.fit_transform(X_static_raw)
    
    # 2. One-Hot Encode Categorical Soil Data
    lb = LabelBinarizer()
    # The fit_transform call assumes the soil_class column has integers (e.g., HWSD codes)
    X_soil_encoded = lb.fit_transform(X_soil_raw)
    
    # 3. Scale Temporal (Rainfall) Features
    # MinMax scaling is often preferred for time series/LSTM inputs (0 to 1)
    scaler_temporal = MinMaxScaler()
    X_temporal_scaled = scaler_temporal.fit_transform(X_temporal_raw)

    # 4. Combine Static Features
    X_static_final = np.hstack((X_static_scaled, X_soil_encoded))
    
    # 5. Reshape Data for CNN-LSTM
    # Temporal: Must be 3D [samples, timesteps, features]
    # Here, 'features' is 1 (rainfall depth)
    X_temporal_reshaped = X_temporal_scaled.reshape(len(X_temporal_scaled), TIMESTEPS, 1)

    # Static: Must be 2D [samples, features]
    # This will be input into the Dense layer later.
    X_static_reshaped = X_static_final 
    
    # 6. Split Data
    # Split the temporal and static data together to maintain sample alignment
    X_temp_train, X_temp_test, X_stat_train, X_stat_test, Y_train, Y_test = train_test_split(
        X_temporal_reshaped, X_static_reshaped, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f"✅ Data prepared. Training set size: {len(Y_train)}, Test set size: {len(Y_test)}")
    return X_temp_train, X_temp_test, X_stat_train, X_stat_test, Y_train, Y_test

# --- 3. MODEL DEFINITION ---

def build_cnn_lstm_hybrid(temporal_shape, static_shape):
    """Defines the CNN-LSTM Hybrid model architecture."""
    from tensorflow.keras.layers import Input, Concatenate
    from tensorflow.keras.models import Model
    
    print("🛠️ Building CNN-LSTM Hybrid Model...")

    # --- 1. Temporal Input (Rainfall Sequence) ---
    input_temporal = Input(shape=temporal_shape, name='temporal_input')
    
    # CNN Layer for feature extraction from the sequence
    x = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(input_temporal)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)
    
    # LSTM Layer for sequence modeling
    x = LSTM(64, return_sequences=False)(x) 
    x = Dropout(0.3)(x)
    temporal_output = Dense(16, activation='relu')(x)

    # --- 2. Static Input (Topo/Soil Features) ---
    input_static = Input(shape=(static_shape,), name='static_input')
    
    # Simple Dense layers for static feature processing
    y = Dense(32, activation='relu')(input_static)
    static_output = Dropout(0.3)(y)

    # --- 3. Merger and Output ---
    merged = Concatenate()([temporal_output, static_output])
    
    # Final dense layers
    z = Dense(32, activation='relu')(merged)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='sigmoid')(z) # Sigmoid for binary classification

    model = Model(inputs=[input_temporal, input_static], outputs=output)
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    print("Model Summary:")
    model.summary()
    return model

# --- 4. TRAINING AND EVALUATION ---

def train_and_evaluate(model, X_temp_train, X_temp_test, X_stat_train, X_stat_test, Y_train, Y_test):
    """Trains the model and evaluates performance metrics."""
    
    print("\n🚀 Starting model training...")
    
    # Set up early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [X_temp_train, X_stat_train], 
        Y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 1. Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    # 2. Evaluate the model on the test set
    print("\n🔬 Evaluating model performance...")
    Y_pred_proba = model.predict([X_temp_test, X_stat_test]).flatten()
    Y_pred = (Y_pred_proba > 0.5).astype(int)
    
    # Calculate Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred_proba)
    cm = confusion_matrix(Y_test, Y_pred)
    
    results = f"""
    --- Landslide Susceptibility Model Performance ---
    
    Dataset Size (Test Set): {len(Y_test)} samples
    
    Metrics:
    - Accuracy:  {accuracy:.4f}
    - Precision: {precision:.4f} (Ability to avoid False Positives)
    - Recall:    {recall:.4f} (Ability to find all True Positives / Landslides)
    - F1-Score:  {f1:.4f} (Harmonic mean of Precision and Recall)
    - AUC-ROC:   {auc:.4f} (Measure of separability)
    
    Confusion Matrix:
    [[True Negatives, False Positives]
     [False Negatives, True Positives]]
    {cm}
    """
    
    # 3. Save results to a text file
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        f.write(results)
        
    print(results)
    print(f"✅ Evaluation results saved to {RESULTS_PATH}")


# --- 5. MAIN EXECUTION ---

def main():
    """Orchestrates the model training and evaluation workflow."""
    
    # 1. Load and prepare data
    X_temp_train, X_temp_test, X_stat_train, X_stat_test, Y_train, Y_test = load_and_prepare_data()
    
    if X_temp_train is None:
        return # Exit if data loading failed

    # 2. Define the Model
    # Temporal shape: (TIMESTEPS, 1) -> (7, 1)
    temporal_shape = (X_temp_train.shape[1], X_temp_train.shape[2]) 
    # Static shape: (N_STATIC) -> (e.g., 4 static + 10 soil classes = 14)
    static_shape = X_stat_train.shape[1] 
    
    model = build_cnn_lstm_hybrid(temporal_shape, static_shape)
    
    # 3. Train and Evaluate
    train_and_evaluate(model, X_temp_train, X_temp_test, X_stat_train, X_stat_test, Y_train, Y_test)
    
    print("\n--- Script 03 Completed Successfully ---")

if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()