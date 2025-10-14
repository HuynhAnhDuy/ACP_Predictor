import numpy as np
import torch
import esm
import joblib
import json
from tensorflow.keras.models import load_model

# ====== One-hot encoding ======
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_sequence(seq, max_length):
    """Encode a single peptide sequence as one-hot matrix."""
    encoded = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)
    for j, aa in enumerate(seq[:max_length]):
        if aa in AA_TO_INDEX:
            encoded[j, AA_TO_INDEX[aa]] = 1.0
    return encoded


# ====== ESM Embedding ======
def compute_esm_embedding(seq, model, alphabet, device):
    """Compute mean-pooled ESM embedding for a single sequence."""
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        embedding = token_representations.mean(dim=1).squeeze().cpu().numpy()
    return embedding


# ====== Feature Preparation ======
def prepare_features(sequence, max_length, esm_model, alphabet, device):
    """Combine one-hot and ESM embeddings into joint feature tensor."""
    onehot = one_hot_encode_sequence(sequence, max_length)
    esm_embed = compute_esm_embedding(sequence, esm_model, alphabet, device)
    esm_expanded = np.repeat(esm_embed[np.newaxis, :], max_length, axis=0)
    conjoint = np.concatenate([onehot, esm_expanded], axis=-1)
    return conjoint


# ====== Prediction Function ======
def predict_acp(sequence):
    """Predict ACP probability using CNN + Transformer + RF ensemble."""

    # --- Load model configs ---
    with open("models/config.json", "r") as f:
        config = json.load(f)
    max_length = config["max_length"]
    feature_dim = config["feature_dim"]

    # --- Load trained models ---
    cnn_model = load_model("models/cnn_model.keras")
    transformer_model = load_model("models/transformer_model.keras")
    meta_model = joblib.load("models/meta_rf_model.pkl")

    # --- Load ESM model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
    esm_model = esm_model.to(device)
    esm_model.eval()

    # --- Prepare feature tensor ---
    conjoint = prepare_features(sequence, max_length, esm_model, alphabet, device)
    X_input = conjoint[np.newaxis, :, :]

    # --- Base model predictions ---
    cnn_pred = cnn_model.predict(X_input, verbose=0).flatten()[0]
    trans_pred = transformer_model.predict(X_input, verbose=0).flatten()[0]

    # --- Meta-model final prediction ---
    meta_X = np.array([[cnn_pred, trans_pred]])
    final_prob = meta_model.predict_proba(meta_X)[0, 1]

    # --- Return clean result ---
    label = "ACP-Positive" if final_prob >= 0.5 else "ACP-Negative"
    return {"Sequence": sequence, "Probability": float(final_prob), "Prediction": label}


# ====== CLI Test ======
if __name__ == "__main__":
    print("=== Anti-Cancer Peptide Prediction (Stacking Ensemble) ===")
    seq = input("🧬 Enter peptide sequence: ").strip().upper()
    result = predict_acp(seq)

    print("\n✅ Prediction Result:")
    print(f"Sequence: {result['Sequence']}")
    print(f"Probability: {result['Probability']:.4f}")
    print(f"Prediction: {result['Prediction']}")
