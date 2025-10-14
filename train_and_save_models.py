import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)
from sklearn.ensemble import RandomForestClassifier

# =========================== CNN ===========================
def create_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ========================= Transformer =========================
def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(0.1)(attn_output)
    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(0.1)(ffn)
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    x = GlobalAveragePooling1D()(out2)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================== Training ========================
def train_and_save_models(X_train, y_train):
    os.makedirs("models", exist_ok=True)
    input_shape = X_train.shape[1:]

    print("🚀 Training CNN...")
    cnn = create_cnn(input_shape)
    cnn.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
    cnn.save("models/cnn_model.keras")
    print("✅ Saved: models/cnn_model.keras")

    print("🚀 Training Transformer...")
    transformer = create_transformer(input_shape)
    transformer.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
    transformer.save("models/transformer_model.keras")
    print("✅ Saved: models/transformer_model.keras")

    print("🧠 Getting base model outputs...")
    cnn_preds = cnn.predict(X_train, verbose=0).flatten()
    trans_preds = transformer.predict(X_train, verbose=0).flatten()
    meta_X = np.column_stack([cnn_preds, trans_preds])

    print("🌲 Training Random Forest meta-model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf_model.fit(meta_X, y_train)
    joblib.dump(rf_model, "models/rf_model.pkl")
    print("✅ Saved: models/rf_model.pkl")

# =========================== MAIN ===========================
def main():
    print("📥 Loading conjoint features...")
    X_train_df = pd.read_csv("ACP_x_train_onehot_esm.csv", index_col=0)
    y_train = pd.read_csv("ACP_y_train.csv")["Label"].values

    for col in ["Sequence", "sequence"]:
        if col in X_train_df.columns:
            X_train_df = X_train_df.drop(columns=[col])

    X_train_df = X_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_np = X_train_df.values.astype(np.float32)

    if X_np.shape[1] != 2540:
        raise ValueError(f"❌ Expected 2540 features (2220 one-hot + 320 ESM), but got {X_np.shape[1]}")

    print("✅ Splitting one-hot and ESM features...")

    X_onehot = X_np[:, :2220].reshape(-1, 111, 20)
    X_esm = X_np[:, 2220:]  # shape (N, 320)
    X_esm_expanded = np.repeat(X_esm[:, np.newaxis, :], 111, axis=1)  # shape (N, 111, 320)

    X_total = np.concatenate([X_onehot, X_esm_expanded], axis=-1)  # shape (N, 111, 340)

    print(f"✅ Final shape: {X_total.shape} (should be (N, 111, 340))")

    train_and_save_models(X_total, y_train)

if __name__ == "__main__":
    main()
