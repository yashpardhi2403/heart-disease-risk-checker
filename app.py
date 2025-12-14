# import streamlit as st
# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier


# # ---------------------------
# # 1. LOAD DATA
# # ---------------------------

# @st.cache_data
# def load_data():
#     df = pd.read_csv("heart_2020_cleaned.csv")
#     df = df.dropna()
#     return df


# # ---------------------------
# # 2. TRAIN MODEL + PREPROCESSING OBJECTS``
# # ---------------------------

# @st.cache_resource
# def train_model(df):
#     target = "HeartDisease"

#     # Separate features and target
#     X_raw = df.drop(columns=[target])
#     y = df[target]

#     # One-hot encode all categoricals
#     X_encoded = pd.get_dummies(X_raw)

#     # Save feature column names for alignment later
#     feature_cols = X_encoded.columns.tolist()

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_encoded, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # Detect numeric columns in original df
#     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     if target in numeric_cols:
#         numeric_cols.remove(target)

#     # Scale numeric columns
#     scaler = StandardScaler()
#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()

#     X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#     X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

#     # Train best model (Random Forest)
#     best_model = RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         n_jobs=-1
#     )
#     best_model.fit(X_train_scaled, y_train)

#     return best_model, scaler, feature_cols, numeric_cols, X_raw.columns.tolist()


# def preprocess_single(sample_dict, feature_cols, scaler, numeric_cols):
#     # dict -> DataFrame
#     df_single = pd.DataFrame([sample_dict])

#     # One-hot encode with same logic as training
#     df_single_enc = pd.get_dummies(df_single)

#     # Align columns with training (missing = 0)
#     df_single_enc = df_single_enc.reindex(columns=feature_cols, fill_value=0)

#     # Scale numeric features
#     df_single_enc[numeric_cols] = scaler.transform(df_single_enc[numeric_cols])

#     return df_single_enc


# def predict_single_person(sample_dict, model, scaler, feature_cols, numeric_cols):
#     X_single = preprocess_single(sample_dict, feature_cols, scaler, numeric_cols)

#     pred_class = model.predict(X_single)[0]

#     # Probability for positive class ("Yes")
#     classes = list(model.classes_)
#     if "Yes" in classes:
#         pos_idx = classes.index("Yes")
#     else:
#         pos_idx = 1 if len(classes) > 1 else 0

#     pred_proba = model.predict_proba(X_single)[0][pos_idx]

#     return pred_class, float(pred_proba)


# # ---------------------------
# # 3. STREAMLIT UI
# # ---------------------------

# def main():
#     st.title("🫀 Heart Disease Risk Checker (BRFSS 2020)")
#     st.write("Enter your health details to estimate the risk of heart disease using a trained ML model.")

#     # Load data and train model
#     df = load_data()
#     model, scaler, feature_cols, numeric_cols, feature_cols_raw = train_model(df)

#     st.sidebar.header("About the App")
#     st.sidebar.markdown(
#         """
#         - Dataset: **CDC BRFSS 2020 (Heart Disease subset)**
#         - Model: **Random Forest Classifier**
#         - Target: **HeartDisease (Yes/No)**
#         """
#     )

#     st.subheader("🔢 Enter Your Details")

#     # We use the original feature columns from X_raw (not encoded)
#     X_raw = df[feature_cols_raw]

#     sample = {}

#     # Build input widgets dynamically based on column types
#     for col in feature_cols_raw:
#         col_data = X_raw[col]
#         if pd.api.types.is_numeric_dtype(col_data):
#             # Numeric input
#             min_val = float(col_data.min())
#             max_val = float(col_data.max())
#             default_val = float(col_data.median())

#             sample[col] = st.number_input(
#                 f"{col}",
#                 min_value=min_val,
#                 max_value=max_val,
#                 value=default_val
#             )
#         else:
#             # Categorical input - use existing unique values from dataset
#             options = sorted(col_data.dropna().unique().tolist())
#             default_val = options[0] if options else ""
#             sample[col] = st.selectbox(f"{col}", options, index=0)

#     if st.button("Predict Heart Disease Risk"):
#         pred_label, pred_prob = predict_single_person(
#             sample, model, scaler, feature_cols, numeric_cols
#         )

#         st.markdown("---")
#         st.subheader("🧾 Prediction Result")

#         if pred_label == "Yes":
#             st.error(f"High chance of heart disease 😟")
#         else:
#             st.success(f"Low chance of heart disease 🙂")

#         st.write(f"**Estimated probability of Heart Disease (Yes):** `{pred_prob:.3f}`")

#         st.markdown("### Your Input Summary")
#         st.json(sample)


# if __name__ == "__main__":
#     main()



# import streamlit as st
# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# import shap
# from openai import OpenAI

# client = OpenAI()  # uses OPENAI_API_KEY env variable


# # ---------------------------
# # 1. LOAD DATA
# # ---------------------------

# @st.cache_data
# def load_data():
#     df = pd.read_csv("heart_2020_cleaned.csv")
#     df = df.dropna()
#     return df


# # ---------------------------
# # 2. TRAIN MODEL + OBJECTS
# # ---------------------------

# @st.cache_resource
# def train_model(df):
#     target = "HeartDisease"

#     X_raw = df.drop(columns=[target])
#     y = df[target]

#     # one-hot encoding
#     X_encoded = pd.get_dummies(X_raw)
#     feature_cols = X_encoded.columns.tolist()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_encoded, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # numeric cols from original df
#     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     if target in numeric_cols:
#         numeric_cols.remove(target)

#     scaler = StandardScaler()
#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()

#     X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#     X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

#     # Random Forest as best model
#     best_model = RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         n_jobs=-1
#     )
#     best_model.fit(X_train_scaled, y_train)

#     # SHAP explainer (tree-based)
#     explainer = shap.TreeExplainer(best_model)

#     return best_model, scaler, feature_cols, numeric_cols, X_raw.columns.tolist(), explainer


# def preprocess_single(sample_dict, feature_cols, scaler, numeric_cols):
#     df_single = pd.DataFrame([sample_dict])
#     df_single_enc = pd.get_dummies(df_single)
#     df_single_enc = df_single_enc.reindex(columns=feature_cols, fill_value=0)
#     df_single_enc[numeric_cols] = scaler.transform(df_single_enc[numeric_cols])
#     return df_single_enc


# def predict_single_person(sample_dict, model, scaler, feature_cols, numeric_cols):
#     X_single = preprocess_single(sample_dict, feature_cols, scaler, numeric_cols)

#     pred_class = model.predict(X_single)[0]

#     classes = list(model.classes_)
#     if "Yes" in classes:
#         pos_idx = classes.index("Yes")
#     else:
#         pos_idx = 1 if len(classes) > 1 else 0

#     pred_proba = model.predict_proba(X_single)[0][pos_idx]

#     return pred_class, float(pred_proba), X_single


# # ---------------------------
# # 3. GPT HEALTH RECOMMENDATION
# # ---------------------------

# def generate_health_advice(sample, pred_label, pred_prob):
#     """
#     Uses GPT to generate simple, non-diagnostic lifestyle advice
#     based on input features and predicted risk.
#     """
#     risk_text = "high" if pred_label == "Yes" or pred_prob >= 0.5 else "low"

#     prompt = f"""
# You are a helpful health assistant (not a doctor). 
# A non-clinical heart disease risk model predicted a {risk_text} chance of heart disease
# with probability {pred_prob:.2f} for this person based on these factors:

# {sample}

# Based on general public-health guidance only, give 5–7 short, practical tips 
# about lifestyle (diet, exercise, smoking, sleep, alcohol, stress, regular checkups).
# Avoid medical jargon, do NOT give medication names, and clearly say that this 
# is not a diagnosis and they should consult a doctor for personal medical advice.
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system", "content": "You give safe, general lifestyle guidance only."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.6,
#             max_tokens=400,
#         )
#         advice = response.choices[0].message.content.strip()
#     except Exception as e:
#         advice = f"Could not generate AI advice right now: {e}"

#     return advice


# # ---------------------------
# # 4. STREAMLIT UI
# # ---------------------------

# def main():
#     st.title("🫀 Heart Disease Risk Checker (BRFSS 2020)")
#     st.write("Enter your health details to estimate the risk of heart disease using a trained ML model.")

#     df = load_data()
#     model, scaler, feature_cols, numeric_cols, feature_cols_raw, explainer = train_model(df)

#     st.sidebar.header("About the App")
#     st.sidebar.markdown(
#         """
#         - Dataset: **CDC BRFSS 2020 (Heart Disease subset)**
#         - Model: **Random Forest Classifier**
#         - Target: **HeartDisease (Yes/No)**
#         - Note: This tool is **not a diagnosis**. Always consult a doctor.
#         """
#     )

#     st.subheader("🔢 Enter Your Details")

#     X_raw = df[feature_cols_raw]
#     sample = {}

#     for col in feature_cols_raw:
#         col_data = X_raw[col]
#         if pd.api.types.is_numeric_dtype(col_data):
#             min_val = float(col_data.min())
#             max_val = float(col_data.max())
#             default_val = float(col_data.median())
#             sample[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val)
#         else:
#             options = sorted(col_data.dropna().unique().tolist())
#             default_index = 0
#             sample[col] = st.selectbox(col, options, index=default_index)

#     if st.button("Predict Heart Disease Risk"):
#         pred_label, pred_prob, X_single = predict_single_person(
#             sample, model, scaler, feature_cols, numeric_cols
#         )

#         st.markdown("---")
#         st.subheader("🧾 Prediction Result")

#         if pred_label == "Yes":
#             st.error(f"Model assessment: **Higher chance of heart disease** 😟")
#         else:
#             st.success(f"Model assessment: **Lower chance of heart disease** 🙂")

#         st.write(f"**Estimated probability of Heart Disease (Yes):** `{pred_prob:.3f}`")

#         # ---------- GPT advice ----------
#         st.markdown("### 🤖 AI Lifestyle Suggestions (not medical advice)")
#         advice = generate_health_advice(sample, pred_label, pred_prob)
#         st.write(advice)

#         # ---------- SHAP explanation (Step D) ----------
#         st.markdown("### 🔍 Why did the model predict this? (SHAP explanation)")
#         st.caption("Shows which features pushed the prediction towards higher or lower risk.")

#         # SHAP for a single instance
#         shap_values = explainer.shap_values(X_single)

#         # For binary RF, shap_values is a list [class0, class1]; take class1
#         if isinstance(shap_values, list):
#             sv = shap_values[1][0]
#         else:
#             sv = shap_values[0]

#         # Convert to Series for sorting
#         sv_series = pd.Series(sv, index=X_single.columns).sort_values(key=np.abs, ascending=False).head(10)

#         st.write("Top contributing features:")
#         st.bar_chart(sv_series)

#         st.markdown("### Your Input Summary")
#         st.json(sample)


# if __name__ == "__main__":
#     main()


# import streamlit as st
# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier

# import shap
# from groq import Groq
# import os


# # ---------------------------
# # 1. LOAD DATA
# # ---------------------------

# @st.cache_data
# def load_data():
#     df = pd.read_csv("heart_2020_cleaned.csv")
#     df = df.dropna()
#     return df


# # ---------------------------
# # 2. TRAIN MODEL + OBJECTS
# # ---------------------------

# @st.cache_resource
# def train_model(df):
#     target = "HeartDisease"

#     X_raw = df.drop(columns=[target])
#     y = df[target]

#     # One-hot encode
#     X_encoded = pd.get_dummies(X_raw)
#     feature_cols = X_encoded.columns.tolist()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_encoded, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # numeric columns from original df
#     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     if target in numeric_cols:
#         numeric_cols.remove(target)

#     scaler = StandardScaler()
#     X_train_scaled = X_train.copy()
#     X_test_scaled = X_test.copy()

#     X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#     X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

#     best_model = RandomForestClassifier(
#         n_estimators=300, random_state=42, n_jobs=-1
#     )
#     best_model.fit(X_train_scaled, y_train)

#     explainer = shap.TreeExplainer(best_model)

#     return best_model, scaler, feature_cols, numeric_cols, X_raw.columns.tolist(), explainer


# def preprocess_single(sample_dict, feature_cols, scaler, numeric_cols):
#     df_single = pd.DataFrame([sample_dict])
#     df_single_enc = pd.get_dummies(df_single)
#     df_single_enc = df_single_enc.reindex(columns=feature_cols, fill_value=0)
#     df_single_enc[numeric_cols] = scaler.transform(df_single_enc[numeric_cols])
#     return df_single_enc


# def predict_single_person(sample_dict, model, scaler, feature_cols, numeric_cols):
#     X_single = preprocess_single(sample_dict, feature_cols, scaler, numeric_cols)

#     pred_class = model.predict(X_single)[0]

#     classes = list(model.classes_)
#     pos_idx = classes.index("Yes") if "Yes" in classes else 1

#     pred_proba = model.predict_proba(X_single)[0][pos_idx]

#     return pred_class, float(pred_proba), X_single


# # ---------------------------
# # 3. FREE LLM HEALTH ADVICE (Groq)
# # ---------------------------

# def generate_health_advice(sample, pred_label, pred_prob):
#     risk_text = "high" if pred_label == "Yes" or pred_prob >= 0.5 else "low"

#     prompt = f"""
#     Provide 5 short lifestyle improvement tips to reduce heart risk.
#     Predicted risk: {risk_text} with probability {pred_prob:.2f}
#     Patient details: {sample}

#     Tips must be simple and safe: diet, exercise, sleep, stress management, quitting smoking, and doctor checkups.
#     Avoid medical claims and medications. Say clearly this is not a diagnosis.
#     """

#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise RuntimeError(
#             "GROQ_API_KEY environment variable is not set. "
#             "Set it before running the app."
#         )

#     client = Groq(api_key=api_key)

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.6,
#         max_tokens=300,
#     )

#     return response.choices[0].message.content.strip()


# # ---------------------------
# # 4. STREAMLIT UI
# # ---------------------------


# import streamlit as st
# import pandas as pd
# import numpy as np

# def main():
#     st.set_page_config(
#         page_title="Heart Disease Risk Checker",
#         page_icon="🫀",
#         layout="centered"
#     )

#     # ---------- Light UI tweaks ----------
#     st.markdown(
#         """
#         <style>
#         .main {
#             background-color: #f7f7f7;
#         }
#         .block-container {
#             padding-top: 2rem;
#             padding-bottom: 2rem;
#             max-width: 900px;
#         }
#         .stButton>button {
#             width: 100%;
#             border-radius: 999px;
#             padding: 0.6rem 1rem;
#             font-weight: 600;
#         }
#         .stAlert {
#             border-radius: 0.75rem;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     st.title("🫀 Heart Disease Risk Checker (BRFSS 2020)")
#     st.write(
#         "This tool uses machine learning to estimate heart disease risk and provide "
#         "AI-based safety recommendations. This is **not** a medical diagnosis."
#     )

#     df = load_data()
#     model, scaler, feature_cols, numeric_cols, feature_cols_raw, explainer = train_model(df)

#     # ---------- Sidebar ----------
#     st.sidebar.header("ℹ About the Tool")
#     st.sidebar.markdown(
#         """
#         **Dataset:** CDC BRFSS 2020  
#         **Model:** Random Forest  
#         **Goal:** Predict likelihood of heart disease  
#         **Disclaimer:** Not a substitute for professional medical advice.
#         """
#     )

#     st.sidebar.markdown("---")
#     st.sidebar.markdown("👨‍⚕️ *If you are worried about your heart health, please consult a doctor.*")

#     # ---------- Input Section ----------
#     st.subheader("🧍 Your Health Details")
#     st.caption("Fill in the fields below. Defaults are set to typical values from the dataset.")

#     X_raw = df[feature_cols_raw]
#     sample = {}

#     # Use a form so user fills everything, then clicks one button
#     with st.form("user_input_form"):
#         col_left, col_right = st.columns(2)

#         for feature in feature_cols_raw:
#             col_data = X_raw[feature]

#             # Prettier label
#             label = feature.replace("_", " ").title()

#             if feature in numeric_cols:
#                 with col_left:
#                     sample[feature] = st.number_input(
#                         label,
#                         min_value=float(col_data.min()),
#                         max_value=float(col_data.max()),
#                         value=float(col_data.median())
#                     )
#             else:
#                 with col_right:
#                     # Keep underlying values same, just show nicely
#                     options = sorted(col_data.unique())
#                     sample[feature] = st.selectbox(
#                         label,
#                         options,
#                         index=options.index(col_data.mode()[0]) if len(col_data.mode()) > 0 else 0
#                     )

#         submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

#     if submitted:
#         pred_label, pred_prob, X_single = predict_single_person(
#             sample, model, scaler, feature_cols, numeric_cols
#         )

#         st.markdown("---")
#         st.subheader("📊 Prediction Result")

#         if pred_label == "Yes":
#             st.error("⚠ Higher chance of Heart Disease")
#         else:
#             st.success("💚 Lower chance of Heart Disease")

#         st.write(f"### Probability: **{pred_prob:.3f}**")

#         # ---------- LLM advice ----------
#         st.markdown("### 🤖 AI Lifestyle Recommendations")
#         try:
#             advice = generate_health_advice(sample, pred_label, pred_prob)
#             st.write(advice)
#         except Exception as e:
#             st.error(f"LLM generation failed: {e}")

#         # ---------- SHAP explanation ----------
#         st.markdown("### 🔍 Feature Contribution (SHAP values)")
#         st.caption("Shows which features pushed the prediction towards higher or lower risk for you.")

#         shap_values = explainer.shap_values(X_single)

#         # Robust handling of different SHAP shapes
#         if isinstance(shap_values, list):
#             sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
#         else:
#             sv = shap_values

#         sv = np.array(sv)

#         # Remove sample dimension if present
#         if sv.ndim == 3:
#             sv = sv[0]
#         elif sv.ndim == 2 and sv.shape[0] == 1:
#             sv = sv[0]

#         # If still 2D, assume (n_features, n_classes)
#         if sv.ndim == 2:
#             if sv.shape[1] > 1:
#                 sv = sv[:, 1]   # positive class
#             else:
#                 sv = sv[:, 0]

#         # Now sv is 1D
#         sv_series = pd.Series(sv, index=X_single.columns).sort_values(
#             key=np.abs, ascending=False
#         ).head(10)

#         st.write("Top contributing features:")
#         st.bar_chart(sv_series)

#         st.markdown("### 📝 Your Input Summary")
#         st.json(sample)
import streamlit as st
import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import shap
from groq import Groq


# ---------------------------
# FEATURE GROUPING (UI MODULARITY)
# ---------------------------

FEATURE_GROUPS = {
    "🧍 Demographics": [
        "Sex", "AgeCategory", "Race"
    ],
    "🫀 Health & Vitals": [
        "BMI", "PhysicalHealth", "MentalHealth", "SleepTime"
    ],
    "🏃 Lifestyle": [
        "Smoking", "AlcoholDrinking", "PhysicalActivity"
    ],
    "🩺 Medical History": [
        "Diabetic", "Asthma", "KidneyDisease", "SkinCancer", "Stroke"
    ]
}


# ---------------------------
# LOAD DATA
# ---------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("heart_2020_cleaned.csv")
    df = df.dropna()
    return df


# ---------------------------
# TRAIN MODEL
# ---------------------------

@st.cache_resource
def train_model(df):
    target = "HeartDisease"

    X_raw = df.drop(columns=[target])
    y = df[target]

    X_encoded = pd.get_dummies(X_raw)
    feature_cols = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    explainer = shap.TreeExplainer(model)

    return model, scaler, feature_cols, numeric_cols, X_raw.columns.tolist(), explainer


# ---------------------------
# PREPROCESS SINGLE INPUT
# ---------------------------

def preprocess_single(sample_dict, feature_cols, scaler, numeric_cols):
    df_single = pd.DataFrame([sample_dict])
    df_single_enc = pd.get_dummies(df_single)
    df_single_enc = df_single_enc.reindex(columns=feature_cols, fill_value=0)
    df_single_enc[numeric_cols] = scaler.transform(df_single_enc[numeric_cols])
    return df_single_enc


def predict_single_person(sample_dict, model, scaler, feature_cols, numeric_cols):
    X_single = preprocess_single(sample_dict, feature_cols, scaler, numeric_cols)
    pred_label = model.predict(X_single)[0]

    classes = list(model.classes_)
    pos_idx = classes.index("Yes")
    pred_prob = model.predict_proba(X_single)[0][pos_idx]

    return pred_label, float(pred_prob), X_single


# ---------------------------
# GROQ LLM ADVICE
# ---------------------------

def generate_health_advice(sample, pred_label, pred_prob):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    risk = "high" if pred_label == "Yes" or pred_prob >= 0.5 else "low"

    prompt = f"""
    Provide 5 short, safe lifestyle tips to reduce heart disease risk.
    Risk level: {risk} (probability {pred_prob:.2f})
    Patient details: {sample}

    Tips should cover diet, exercise, sleep, stress, smoking, and checkups.
    Avoid medicines. Clearly say this is not a diagnosis.
    """

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


# ---------------------------
# STREAMLIT APP
# ---------------------------

def main():
    st.set_page_config(
        page_title="Heart Disease Risk Checker",
        page_icon="🫀",
        layout="centered"
    )

    st.markdown("""
    <style>
    label { font-size: 0.85rem !important; }
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🫀 Heart Disease Risk Checker (BRFSS 2020)")
    st.caption("ML-based risk estimation with AI lifestyle guidance (not a diagnosis).")

    df = load_data()
    model, scaler, feature_cols, numeric_cols, raw_features, explainer = train_model(df)

    X_raw = df[raw_features]
    sample = {}

    st.subheader("🧍 Enter Your Health Details")

    with st.form("input_form"):
        for group, features in FEATURE_GROUPS.items():
            with st.expander(group, expanded=True):
                cols = st.columns(3)
                idx = 0

                for feature in features:
                    data_col = X_raw[feature]
                    label = feature.replace("_", " ")

                    with cols[idx]:
                        if feature in numeric_cols:
                            sample[feature] = st.number_input(
                                label,
                                min_value=float(data_col.min()),
                                max_value=float(data_col.max()),
                                value=float(data_col.median()),
                                step=0.1
                            )
                        else:
                            options = sorted(data_col.unique())
                            sample[feature] = st.selectbox(
                                label,
                                options,
                                index=options.index(data_col.mode()[0])
                            )
                    idx = (idx + 1) % 3

        submitted = st.form_submit_button("🔍 Predict Risk")

    if submitted:
        pred_label, pred_prob, X_single = predict_single_person(
            sample, model, scaler, feature_cols, numeric_cols
        )

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if pred_label == "Yes":
            st.error("⚠ Higher chance of Heart Disease")
        else:
            st.success("💚 Lower chance of Heart Disease")

        st.metric("Probability", f"{pred_prob:.3f}")

        st.subheader("🤖 AI Lifestyle Recommendations")
        try:
            advice = generate_health_advice(sample, pred_label, pred_prob)
            st.write(advice)
        except Exception as e:
            st.warning(f"LLM unavailable: {e}")

        st.subheader("🔍 Feature Contribution (SHAP)")
        shap_values = explainer.shap_values(X_single)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        shap_vals = shap_vals[0]
        top_feats = pd.Series(
            shap_vals, index=X_single.columns
        ).sort_values(key=np.abs, ascending=False).head(10)

        st.bar_chart(top_feats)

        st.subheader("📝 Input Summary")
        st.json(sample)


if __name__ == "__main__":
    main()


# def main():
#     st.set_page_config(
#         page_title="Heart Disease Risk Checker",
#         page_icon="🫀",
#         layout="centered"
#     )

#     # ---------- Light UI Theme ----------
#     st.markdown("""
#         <style>
#             .main { background-color: #fafafa; }
#             .block-container { padding-top: 2rem; max-width: 900px; }
#             label { font-size: 0.85rem !important; font-weight: 600 !important; }
#             .desc-text { font-size: 0.70rem; color: #666; margin-top: -6px; margin-bottom: 6px; }
#             .stButton>button { width: 100%; border-radius: 999px; padding: 0.6rem 1rem; font-weight: 600; }
#         </style>
#     """, unsafe_allow_html=True)

#     # ---------- Title ----------
#     st.title("🫀 Heart Disease Risk Checker (BRFSS 2020)")
#     st.write("This tool uses machine learning to estimate heart disease risk and provide AI-based lifestyle recommendations.\n\n**Not a medical diagnosis.**")

#     # Load dataset & model
#     df = load_data()
#     model, scaler, feature_cols, numeric_cols, feature_cols_raw, explainer = train_model(df)

#     # ---------- Sidebar ----------
#     st.sidebar.header("ℹ About the Tool")
#     st.sidebar.markdown(
#         """
#         **Dataset:** CDC BRFSS 2020  
#         **Model:** Random Forest  
#         **Goal:** Predict likelihood of heart disease  
#         **Disclaimer:** Not a substitute for medical evaluation.
#         """
#     )
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("👨‍⚕️ *Consult a doctor if you have heart-related concerns.*")

#     # ---------- Field Descriptions ----------
#     descriptions = {
#         "BMI": "Body mass index.",
#         "Smoking": "Ever smoked ≥100 cigarettes.",
#         "AlcoholDrinking": "Heavy alcohol use.",
#         "Stroke": "Diagnosed stroke history.",
#         "PhysicalHealth": "Days physical health was poor (last 30).",
#         "MentalHealth": "Days mental health was poor (last 30).",
#         "DiffWalking": "Difficulty walking/climbing stairs.",
#         "Sex": "Biological sex.",
#         "AgeCategory": "Age group.",
#         "Race": "Race / ethnicity.",
#         "Diabetic": "Diagnosed diabetes status.",
#         "PhysicalActivity": "Any exercise outside job.",
#         "GenHealth": "Self-rated health.",
#         "SleepTime": "Hours of sleep per day.",
#         "Asthma": "Diagnosed asthma.",
#         "KidneyDisease": "Diagnosed kidney disease.",
#         "SkinCancer": "Diagnosed skin cancer."
#     }

#     # ---------- Tab Groups ----------
#     basics = ["Sex", "AgeCategory", "Race", "GenHealth", "BMI", "SleepTime"]
#     lifestyle = ["Smoking", "AlcoholDrinking", "PhysicalActivity"]
#     health_mobility = ["PhysicalHealth", "MentalHealth", "DiffWalking"]
#     conditions = ["Diabetic", "Stroke", "Asthma", "KidneyDisease", "SkinCancer"]

#     grouped = set(basics + lifestyle + health_mobility + conditions)
#     leftovers = [f for f in feature_cols_raw if f not in grouped]

#     X_raw = df[feature_cols_raw]
#     sample = {}

#     # ---------- Input Renderer ----------
#     def render_feature(name, container):
#         col_data = X_raw[name]
#         label = name.replace("_", " ").title()
#         desc = descriptions.get(name, "")

#         with container:
#             st.write(f"**{label}**")
#             st.markdown(f"<p class='desc-text'>{desc}</p>", unsafe_allow_html=True)
#             if name in numeric_cols:
#                 sample[name] = st.number_input(
#                     "", float(col_data.min()), float(col_data.max()),
#                     float(col_data.median()), key=name
#                 )
#             else:
#                 opts = sorted(col_data.unique())
#                 default = opts.index(col_data.mode()[0]) if len(col_data.mode()) else 0
#                 sample[name] = st.selectbox("", opts, index=default, key=name)

#     # ---------- Multi-Tab Entry Form ----------
#     st.subheader("🧍 Your Health Details")
#     st.caption("Organized entry with short labels and mini descriptions.")

#     with st.form("input_form"):
#         tab1, tab2, tab3, tab4, tab5 = st.tabs(
#             ["Basics", "Lifestyle", "Health & Mobility", "Conditions", "Other"]
#         )

#         with tab1:
#             c1, c2 = st.columns(2)
#             for i, f in enumerate(basics):
#                 render_feature(f, c1 if i % 2 == 0 else c2)

#         with tab2:
#             c1, c2 = st.columns(2)
#             for i, f in enumerate(lifestyle):
#                 render_feature(f, c1 if i % 2 == 0 else c2)

#         with tab3:
#             c1, c2 = st.columns(2)
#             for i, f in enumerate(health_mobility):
#                 render_feature(f, c1 if i % 2 == 0 else c2)

#         with tab4:
#             c1, c2 = st.columns(2)
#             for i, f in enumerate(conditions):
#                 render_feature(f, c1 if i % 2 == 0 else c2)

#         with tab5:
#             if leftovers:
#                 c1, c2 = st.columns(2)
#                 for i, f in enumerate(leftovers):
#                     render_feature(f, c1 if i % 2 == 0 else c2)
#             else:
#                 st.caption("_No additional fields_")

#         submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

#     # ---------- Prediction Section ----------
#     if submitted:
#         pred_label, pred_prob, X_single = predict_single_person(sample, model, scaler, feature_cols, numeric_cols)

#         st.markdown("---")
#         st.subheader("📊 Prediction Result")
#         st.success("💚 Lower chance of Heart Disease") if pred_label == "No" else st.error("⚠ Higher chance of Heart Disease")

#         st.write(f"### Probability: **{pred_prob:.3f}**")

#         st.markdown("### 🤖 AI Lifestyle Recommendations")
#         try:
#             st.write(generate_health_advice(sample, pred_label, pred_prob))
#         except Exception as e:
#             st.error(f"LLM generation failed: {e}")

#         st.markdown("### 🔍 SHAP Feature Contribution")
#         shap_values = explainer.shap_values(X_single)
#         sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else (
#             shap_values[0] if isinstance(shap_values, list) else shap_values
#         )
#         sv = np.array(sv)
#         if sv.ndim == 3: sv = sv[0]
#         if sv.ndim == 2 and sv.shape[0] == 1: sv = sv[0]
#         if sv.ndim == 2: sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]

#         sv_series = pd.Series(sv, index=X_single.columns).sort_values(key=np.abs, ascending=False).head(10)
#         st.bar_chart(sv_series)

#         st.markdown("### 📝 Your Input Summary")
#         st.json(sample)



if __name__ == "__main__":
    main()

