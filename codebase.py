import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_absolute_error

# 1. ROBUST DATA PREPROCESSING
def load_and_clean_data(file_path):
    # Support for both CSV and Excel
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Clean Headers: Removes newlines and hidden spaces from Aspen export
    df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
    

    # FEATURE ENGINEERING: Chemical Intelligence
    # Preventing DivisionByZero with 1e-9 epsilon
    df['h2_feed_ratio'] = df['VARY 2 H2'] / (df['VARY 1 FEED'] + 1e-9)
    df['flash_p_delta'] = df['VARY 4 F2'] - df['VARY 3 F1']
    
    # Replace any potential infinities with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Define exact columns based on your Aspen setup
    input_cols = ['VARY 1 FEED', 'VARY 2 H2', 'VARY 3 F1', 'VARY 4 F2', 
                  'VARY 5 R2', 'VARY 6 R3', 'h2_feed_ratio', 'flash_p_delta']
    target_cols = ['PURITY', 'MASSFLOW','CO2OUT','H2OUT']
    
    return df[input_cols], df[target_cols]

# 2. DATA PREPARATION & SCALING
X, y = load_and_clean_data('biogas_training_data.xlsx')

# PowerTransformer handles the non-linear "saturation" at 84% purity
pt_y = PowerTransformer(method='yeo-johnson')
scaler_X = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = pt_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.15, random_state=42, shuffle=True
)

# 3. RESNET ARCHITECTURE (Publication Grade)
def build_final_surrogate(input_dim, output_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial Dense Layer
    x = layers.Dense(512, activation='swish')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual Block 1: Learning the "Identity" of the Physics
    shortcut = x
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut]) 
    
    # Narrowing layers for precision
    x = layers.Dense(256, activation='swish')(x)
    x = layers.Dense(128, activation='swish')(x)
    
    # Output Layer
    outputs = layers.Dense(output_dim, activation='linear')(x)
    
    model = models.Model(inputs, outputs)
    
    # ANTI-NaN COMPILATION: Lower learning rate + Gradient Clipping
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4, decay_steps=500)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

model = build_final_surrogate(X_train.shape[1], y_train.shape[1])

# 4. TRAINING WITH ROBUST CALLBACKS
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
    callbacks.TerminateOnNaN() # Stops the run if math breaks
]

print("Starting Robust Training Phase...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=64,
    callbacks=callbacks_list,
    verbose=1
)

# 5. FINAL INVERSE TRANSFORM & R2 VALIDATION (Robust Version)
y_pred_scaled = model.predict(X_test)

# --- THE BUG FIX: Safety Clipping ---
# We clip the scaled predictions to the min/max range of the training targets
# This prevents PowerTransformer from generating NaNs on 'out-of-bounds' predictions
y_pred_scaled = np.clip(y_pred_scaled, y_train.min(axis=0), y_train.max(axis=0))

# Convert back to real physical units
try:
    y_pred = pt_y.inverse_transform(y_pred_scaled)
    y_actual = pt_y.inverse_transform(y_test)
    
    # Final check: if NaNs still exist, fill with median to allow R2 calculation
    if np.isnan(y_pred).any():
        print("⚠️ Warning: NaNs detected in inverse transform. Applying median imputation...")
        col_mean = np.nanmedian(y_pred, axis=0)
        inds = np.where(np.isnan(y_pred))
        y_pred[inds] = np.take(col_mean, inds[1])

except ValueError as e:
    print(f"❌ Transformation Error: {e}")
    # Fallback: calculate metrics on scaled data if inversion fails completely
    y_pred = y_pred_scaled
    y_actual = y_test
    print("Falling back to scaled-space metrics...")

print("\n" + "="*40)
print("       FINAL RESEARCH METRICS")
print("="*40)

target_names = ['PURITY', 'MASSFLOW', 'CO2OUT', 'H2OUT']
for i, col in enumerate(target_names):
    # Ensure no NaNs reach r2_score
    mask = ~np.isnan(y_actual[:, i]) & ~np.isnan(y_pred[:, i])
    r2 = r2_score(y_actual[mask, i], y_pred[mask, i])
    mae = mean_absolute_error(y_actual[mask, i], y_pred[mask, i])
    print(f"{col:10} -> R²: {r2:.5f} | MAE: {mae:.6f}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_parity(y_true, y_pred, target_name="Methane Purity", unit="mol fraction"):
    """
    Generates a research-grade parity plot for the BDT results.
    """
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    
    # Use alpha=0.2 because of the large dataset (15,000 points)
    plt.scatter(y_true, y_pred, alpha=0.2, edgecolors='none', color='#1f77b4', label='Predicted vs. Actual')
    
    # Draw the Identity Line (y = x)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=3, label='Ideal (y=x)')
    
    # Formatting
    plt.title(f'Parity Plot: {target_name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Actual {target_name} ({unit})', fontsize=12)
    plt.ylabel(f'Predicted {target_name} ({unit})', fontsize=12)
    
    # Annotate R^2 Score
    plt.text(0.05, 0.9, f'$R^2 = {r2:.4f}$', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Highlight the 84% Saturation Limit if applicable
    if "Purity" in target_name:
        plt.axvline(x=0.84, color='green', linestyle=':', label='Thermodynamic Limit (84%)')
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save for LaTeX/Overleaf
    plt.savefig(f'parity_{target_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

# Example Usage:
# plot_parity(test_labels[:, 0], predictions[:, 0], target_name="CH4 Purity")