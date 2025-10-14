import numpy as np
import torch
import esm

# Danh sách 20 amino acid chuẩn
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

MAX_LENGTH = 111  # cố định theo mô hình huấn luyện
ESM_DIM = 320     # từ mô hình ESM-2-t6

def get_one_hot(seq: str, max_length: int = MAX_LENGTH) -> np.ndarray:
    encoded = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)
    for j, aa in enumerate(seq[:max_length]):
        if aa in AA_TO_INDEX:
            encoded[j, AA_TO_INDEX[aa]] = 1.0
    return encoded  # (max_length, 20)

def get_esm_embedding(seq: str, model, alphabet, layer: int = 6) -> np.ndarray:
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("sequence", seq)])
    batch_tokens = batch_tokens.to(next(model.parameters()).device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
    token_representations = results["representations"][layer]
    embedding = token_representations[0, 1:len(seq)+1].mean(0).cpu().numpy()
    return embedding  # (320,)

def extract_features(seq: str, esm_model, esm_alphabet) -> np.ndarray:
    seq = seq.upper().strip()

    if len(seq) > MAX_LENGTH:
        print(f"⚠️ Cảnh báo: Sequence quá dài ({len(seq)}), cắt xuống {MAX_LENGTH} ký tự.")
        seq = seq[:MAX_LENGTH]

    onehot = get_one_hot(seq, max_length=MAX_LENGTH)  # (111, 20)
    esm_embed = get_esm_embedding(seq, esm_model, esm_alphabet)  # (320,)
    esm_expanded = np.repeat(esm_embed[np.newaxis, :], MAX_LENGTH, axis=0)  # (111, 320)

    conjoint = np.concatenate([onehot, esm_expanded], axis=-1)  # (111, 340)

    return conjoint.flatten()  # (111 * 340 = 37740,)
