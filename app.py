import streamlit as st

# ==== Page Config ====
st.set_page_config(page_title="Anti-Cancer Peptide Predictor", layout="centered",initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import esm
from tensorflow.keras.models import load_model
from preprocessing import extract_features  # ensure path is correct
from PIL import Image


#==== Global constants ====
MAX_LENGTH = 111
FEATURE_DIM = 340  # 20 one-hot + 320 ESM

# ==== Load models (cache to avoid reload) ====
@st.cache_resource
def load_all_models():
    cnn_model = load_model("models/cnn_model.keras")
    transformer_model = load_model("models/transformer_model.keras")
    rf_model = joblib.load("models/rf_model.pkl")

    # Load ESM model
    esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model = esm_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    return cnn_model, transformer_model, rf_model, esm_model, esm_alphabet

cnn_model, transformer_model, rf_model, esm_model, esm_alphabet = load_all_models()

# ==== Prediction function ====
def predict_peptide(seq, max_length=111):
    seq = seq.strip().upper()

    if len(seq) > max_length:
        st.warning(f"⚠️ Sequence length ({len(seq)}) > {max_length}. It will be truncated.")
        seq = seq[:max_length]

    features = extract_features(seq, esm_model, esm_alphabet)  # shape: (111 * 340,)
    feature_dim = 340
    features = features.reshape(1, max_length, feature_dim)  # (1, 111, 340)

    cnn_out = cnn_model.predict(features, verbose=0).flatten()
    trans_out = transformer_model.predict(features, verbose=0).flatten()
    meta_X = np.column_stack([cnn_out, trans_out])
    prob = rf_model.predict_proba(meta_X)[0][1]
    return prob

# ==== Visit counter ====
def increment_visit_counter():
    counter_file = "visit_count.txt"
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("1")
        return 1
    else:
        with open(counter_file, "r+") as f:
            count = int(f.read().strip())
            count += 1
            f.seek(0)
            f.write(str(count))
            f.truncate()
        return count

visit_count = increment_visit_counter()

# ==== Custom CSS ====
st.markdown("""
    <style>
        /* Make sidebar fixed and scrollable */
        section[data-testid="stSidebar"] {
            background-color: #f5f5f5;
            padding: 1rem;
        }

        /* Style main header */
        h1 {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 2.2rem;
            color: #8B0000;
            text-align: center;
        }

        /* Round button */
        .stButton > button {
            border-radius: 10px;
            background-color: #8B0000;
            color: white;
        }

        /* Prediction result formatting */
        .result-box {
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #8B0000;
            margin-top: 1rem;
        }

        /* Reduce padding between widgets */
        .block-container {
            padding-top: 1rem;
        }

        /* Style tabs */
        div[role="tablist"] > button {
            font-weight: bold;
            font-size: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==== Header ====
st.markdown('<h1 style="color:darkred;">🧬 Anti-Cancer Peptide Predictor (ACPredictor)</h1>', unsafe_allow_html=True)
st.markdown("""
**Developers:** Huynh Anh Duy<sup>1,2</sup>, Tarapong Srisongkram<sup>2</sup>  
**Affiliations:** <sup>1</sup>Can Tho University, Vietnam; <sup>2</sup>Khon Kaen University, Thailand
""", unsafe_allow_html=True)

# ==== About the Tool ====
st.markdown("---")
st.subheader("🔬 About the Tool")
st.markdown("""
    🧪 Predict the **anticancer potential of peptides** using a stacking-based ensemble framework.  
    🧠 This tool leverages a **Convolutional Neural Network (CNN)** and a **Transformer** as base learners, intergrated with a **Random Forest** as the meta-classifier.  
    ➕ The model is trained on **conjoint representations** of peptide sequences, combining **One-hot encoding** and **ESM (Evolutionary Scale Modeling)** embeddings.
    """)

# ==== Sidebar ====
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Input a **peptide sequence** (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).  
    2. Or upload a **CSV file** with a `Sequence` column.  
    3. Click **Predict** for one peptide or **Run batch prediction** for multiple.
    """)
    st.markdown("---")
    st.markdown("""
    🧠 **Prediction Rule:**  
    - **Probability > 0.5** → 🧬 **Likely Anti-Cancer Peptide (ACP)**  
    - **Probability ≤ 0.5** → ⚪ **Non-ACP**
    """)

# ==== Tabs ====
tab1, tab2 = st.tabs(["🧬 Single Prediction", "📁 Batch via CSV"])

# ==== Tab 1: Single Prediction ====
with tab1:
    seq_input = st.text_input("👉 Enter a peptide sequence:", placeholder="e.g. ACDEFGHIKLMNPQRSTVWY")

    if st.button("Predict ACP"):
        if seq_input.strip():
            with st.spinner("🔬 Analyzing peptide..."):
                try:
                    prob = predict_peptide(seq_input)
                    label = "🧬 Likely Anti-Cancer Peptide" if prob > 0.5 else "⚪ Non-ACP"
                    st.subheader("🔎 Prediction Result")
                    st.write(f"**Input sequence:** `{seq_input.strip().upper()}`")
                    st.metric("Predicted Probability", f"{prob:.4f}")
                    st.markdown("---")
                    if prob > 0.5:
                        st.success(label)
                    else:
                        st.info(label)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a valid peptide sequence!")

# ==== Tab 2: Batch Prediction ====
with tab2:
    uploaded_file = st.file_uploader("📤 Upload a CSV file containing a 'Sequence' column", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        if "Sequence" not in df_input.columns:
            st.error("⚠️ The uploaded CSV file must contain a column named 'Sequence'.")
        else:
            st.success(f"✅ Loaded {len(df_input)} peptide sequences.")
            if st.button("🔍 Run Batch Prediction"):
                results = []
                with st.spinner("⏳ Running predictions..."):
                    for seq in df_input["Sequence"]:
                        try:
                            prob = predict_peptide(seq)
                            pred_label = "ACP" if prob > 0.5 else "Non-ACP"
                            results.append({
                                "Sequence": seq,
                                "Predicted_Probability": round(prob, 4),
                                "Prediction": pred_label
                            })
                        except Exception as e:
                            results.append({
                                "Sequence": seq,
                                "Predicted_Probability": None,
                                "Prediction": f"Error: {e}"
                            })
                df_result = pd.DataFrame(results)
                st.dataframe(df_result)

                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv,
                    file_name="acp_predictions.csv",
                    mime="text/csv"
                )
# === Author Section ===
st.markdown("---")
st.subheader("👨‍🔬 About the Authors")

col1, col2 = st.columns(2)

with col1:
    image1 = Image.open("assets/duy.jpg")
    st.image(image1, caption="Huynh Anh Duy", width=160)
    st.markdown("""
    **Huynh Anh Duy, MPharm.**  
    Can Tho University, Vietnam  
    PhD Candidate, Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [huynhanhduy.h@kkumail.com](mailto:huynhanhduy.h@kkumail.com), [haduy@ctu.edu.vn](mailto:haduy@ctu.edu.vn)
    """)

with col2:
    image2 = Image.open("assets/tarasi.png")
    st.image(image2, caption="Tarapong Srisongkram", width=160)
    st.markdown("""
    **Asst Prof. Dr. Tarapong Srisongkram**  
    Faculty of Pharmaceutical Sciences  
    Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [tarasri@kku.ac.th](mailto:tarasri@kku.ac.th)
    """)

# ==== Footer ====
st.markdown("---")
st.caption(f"🧩 Python version: {sys.version.split()[0]}")
st.markdown(f"👁️ **Total visits:** {visit_count}")
st.markdown(
    """
    <div style='text-align:center; font-size:14px; color:gray; line-height:1.6;'>
    ⚠️ <b>Disclaimer:</b> This platform is for <i>research purposes only</i>.  
    The ACP predictions are experimental and not validated for clinical use.  
    Users should interpret the results cautiously. <br><br>
    🧬 <b>Model:</b> CNN + Transformer + RandomForest stacking using One-hot + ESM conjoint features. <br>
    📄 <b>Version:</b> 1.0.0 &nbsp; | &nbsp; <b>Created:</b> October 2025 <br>
    © 2025 QSARLab Research Group
    </div>
    """,
    unsafe_allow_html=True
)
