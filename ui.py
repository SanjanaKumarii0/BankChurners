import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --------- Load Model ---------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --------- Page Config ---------
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")
# --------- Solid Blue Background CSS ---------
page_bg = """
<style>
.stApp {
    background-color: #1E3A8A; /* Solid Blue */
    color: white;
}
.stButton button {
    background-color: #3B82F6;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.5rem 1.5rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --------- App Title ---------
st.title("🏦 Bank Churn Prediction App")
st.write("Predict whether a customer will churn using Naive Bayes model.")

# --------- Input Option ---------
st.sidebar.header("⚙️ Input Options")
input_method = st.sidebar.radio("Choose Input Method", ["Manual Entry", "Upload CSV/Excel"])

# --------- Manual Input (Compact Layout) ---------
if input_method == "Manual Entry":
    st.subheader("🔢 Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, step=1)
        balance = st.number_input("Balance", min_value=0.0, step=100.0, format="%.2f")

    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
        is_active_member = st.selectbox("Active Member?", ["No", "Yes"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, format="%.2f")

    # Create feature vector (dummy 45 features)
    # Replace this mapping with actual dataset feature order
    inputs = np.zeros(45)
    inputs[0] = credit_score
    inputs[1] = age
    inputs[2] = tenure
    inputs[3] = balance
    inputs[4] = num_products
    inputs[5] = 1 if has_cr_card == "Yes" else 0
    inputs[6] = 1 if is_active_member == "Yes" else 0
    inputs[7] = estimated_salary

# --------- File Upload Input ---------
else:
    st.subheader("📂 Upload Customer Data (CSV/Excel)")
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Preview of Data:", df.head())
        inputs = df.values[0]   # take first row as example
    else:
        inputs = None

# --------- Prediction ---------
if st.button("🔮 Predict Churn") and inputs is not None:
    input_data = np.array([inputs])   # ensure 2D

    prediction = model.predict(input_data)
    result = "🚪 Customer is likely to CHURN" if prediction[0] == 1 else "✅ Customer will STAY"

    st.markdown(f"<h2 style='color:#00ffcc;'>{result}</h2>", unsafe_allow_html=True)
