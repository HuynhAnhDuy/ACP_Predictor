import numpy as np
import pandas as pd
import torch
import esm
import joblib
from tqdm import tqdm

# ===== One-hot encoding =====
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode(sequences, max_length):
    num_samples = len(sequences)
    encoded = np.zeros((num_samples, max_length, len(AMINO_ACIDS)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_length]):
            if aa in AA_TO_INDEX:
                encoded[i, j, AA_TO_INDEX[aa]] = 1.0
    return encoded

# ===== ESM embedding (mean representation) =====
def compute_esm_embeddings(sequences, model_name="esm2_t6_8M_UR50D"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    embeddings = []
    model.eval()
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Computing ESM embeddings"):
            batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            sequence_embedding = token_representations.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(sequence_embedding)

    return np.array(embeddings)

# ===== Combine everything =====
def create_conjoint_features(input_csv, output_pkl, output_csv, model_name="esm2_t6_8M_UR50D"):
    df = pd.read_csv(input_csv)
    if 'Sequence' not in df.columns:
        raise ValueError("❌ CSV must contain a 'Sequence' column.")
    sequences = df['Sequence'].tolist()
    max_length = max(len(s) for s in sequences)

    print(f"📏 Max sequence length: {max_length}")
    print("🧬 Encoding One-hot...")
    onehot = one_hot_encode(sequences, max_length)

    print("🧠 Computing ESM embeddings...")
    esm_embed = compute_esm_embeddings(sequences, model_name=model_name)

    # Repeat ESM embedding along max_length to match one-hot shape for concatenation
    esm_expanded = np.repeat(esm_embed[:, np.newaxis, :], max_length, axis=1)

    print("🔗 Concatenating features...")
    conjoint = np.concatenate([onehot, esm_expanded], axis=-1)  # shape: (n_samples, max_length, 20+embedding_dim)

    print(f"💾 Saving to: {output_pkl}")
    joblib.dump(conjoint, output_pkl)

    print(f"💾 Saving to CSV: {output_csv}")
    num_samples, max_len, feat_dim = conjoint.shape
    flat_conjoint = conjoint.reshape(num_samples, -1)
    feature_names = [f"f{j}_{k}" for j in range(max_len) for k in range(feat_dim)]
    df_out = pd.DataFrame(flat_conjoint, columns=feature_names)
    df_out.to_csv(output_csv, index=False)

    print("✅ Done.")

# ===== Main usage =====
if __name__ == "__main__":
    input_csv = "ACP_x_train.csv"             # input file must contain 'Sequence' column
    output_pkl = "ACP_x_train_conjoint.pkl"   # output for model input or webserver
    output_csv = "ACP_x_train_conjoint.csv"   # optional output for inspection or ML tools

    create_conjoint_features(input_csv, output_pkl, output_csv)
