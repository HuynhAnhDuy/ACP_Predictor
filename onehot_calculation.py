import numpy as np
import pandas as pd

# Step 1: Define the 20 standard amino acids and mapping
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Step 2: One-hot encoding function (returns 3D tensor: [samples, max_length, 20])
def one_hot_encode(sequences, max_length):
    num_samples = len(sequences)
    encoded = np.zeros((num_samples, max_length, len(AMINO_ACIDS)), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j < max_length and aa in AA_TO_INDEX:
                encoded[i, j, AA_TO_INDEX[aa]] = 1.0

    return encoded

# Step 3: Generate feature names
def generate_feature_names(max_length):
    feature_names = []
    for pos in range(1, max_length + 1):
        for aa in AMINO_ACIDS:
            feature_names.append(f'pos{pos}_{aa}')
    return feature_names

# Step 4: Process a CSV file and return one-hot encoded DataFrame (flattened to 2D with feature names)
def process_sequences(input_csv, max_length):
    df = pd.read_csv(input_csv)
    if 'Sequence' not in df.columns:
        raise ValueError("Input CSV must contain a 'Sequence' column.")

    sequences = df['Sequence'].tolist()
    onehot_tensor = one_hot_encode(sequences, max_length)

    # Flatten each 2D matrix (max_length x 20) into a 1D vector
    flattened = onehot_tensor.reshape((len(sequences), -1))

    # Generate feature names
    feature_names = generate_feature_names(max_length)

    # Convert to DataFrame with feature names
    onehot_df = pd.DataFrame(flattened, columns=feature_names)
    return onehot_df

# Step 5: Main workflow
if __name__ == "__main__":
    # Input and output file paths
    x_train_file = "ACP_clinical_x_train.csv"
    x_test_file = "ACP_approved_new.csv"
    x_train_output = "ACP_clinical_x_train_onehot.csv"
    x_test_output = "ACP_approved_new_onehot.csv"

    # Step 5.1: Load both files and determine max peptide length
    train_df = pd.read_csv(x_train_file)
    test_df = pd.read_csv(x_test_file)

    train_max_length = train_df['Sequence'].apply(len).max()
    test_max_length = test_df['Sequence'].apply(len).max()
    max_length = max(train_max_length, test_max_length)

    print(f"Max peptide sequence length across both datasets: {max_length}")

    # Step 5.2: One-hot encode and save with feature names
    x_train_onehot = process_sequences(x_train_file, max_length)
    x_test_onehot = process_sequences(x_test_file, max_length)

    x_train_onehot.to_csv(x_train_output, index=False)
    x_test_onehot.to_csv(x_test_output, index=False)

    print(f"One-hot encoded training data saved to: {x_train_output}")
    print(f"One-hot encoded testing data saved to: {x_test_output}")
