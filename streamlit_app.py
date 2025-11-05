# =========================================================
# EMIPredictAI - Final Streamlit App (Stable SHAP + Smart EMI Logic + PDF Report + Logo + Encoding Fix)
# Developed by: Pruthviraj Tarode | MGM’s College of Engineering, Nanded
# =========================================================

import streamlit as st
import pandas as pd
import joblib, os, shap, matplotlib.pyplot as plt, base64
import numpy as np
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EMIPredictAI", page_icon="💰", layout="centered")

# =========================================================
# HEADER SECTION (Loads MGM Logo Automatically)
# =========================================================
base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logo_path = os.path.join(base, "assets", "college_logo.png")

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

if os.path.exists(logo_path):
    logo_b64 = get_base64_image(logo_path)
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="140" style="border-radius:50%; box-shadow:0 4px 12px rgba(0,0,0,0.2);">'
else:
    logo_html = "<div style='color:red;'>[Logo not found in assets folder]</div>"

st.markdown(
    f"""
    <div style="text-align:center; background:linear-gradient(90deg,#e6f2ff,#ffffff); padding:15px; border-radius:12px;">
        {logo_html}
        <h2 style='color:#0078d4;'>🏫 MGM’s College of Engineering, Nanded</h2>
        <h4>Department of Computer Science & Engineering</h4>
        <h3 style='color:#0078d4;'>💰 Project Title: EMIPredictAI</h3>
        <h5>👨‍💻 Developed by: Pruthviraj Tarode (B.Tech CSE)</h5>
    </div>
    <hr style="height:3px; background-color:#0078d4; border:none; border-radius:2px;">
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODELS
# =========================================================
MODEL_C = os.path.join(base, "models", "best_class_model.joblib")
MODEL_R = os.path.join(base, "models", "best_reg_model.joblib")

clf = joblib.load(MODEL_C) if os.path.exists(MODEL_C) else None
reg = joblib.load(MODEL_R) if os.path.exists(MODEL_R) else None

# =========================================================
# INPUT SECTION
# =========================================================
st.sidebar.header("📋 Applicant Details")
age = st.sidebar.number_input("Age", 18, 70, 30)
monthly_salary = st.sidebar.number_input("Monthly Salary (INR)", 1000, 2000000, 50000)
current_emi_amount = st.sidebar.number_input("Current EMI (INR)", 0, 200000, 2000)
other_monthly_expenses = st.sidebar.number_input("Other Monthly Expenses (INR)", 0, 200000, 5000)
existing_loans = st.sidebar.number_input("Existing Loans", 0, 20, 0)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 720)
requested_amount = st.sidebar.number_input("Requested Loan Amount (INR)", 1000, 5000000, 100000)

affordability_ratio = monthly_salary / (requested_amount if requested_amount else 1)
debt_to_income = (current_emi_amount + other_monthly_expenses) / (monthly_salary if monthly_salary else 1)

data = {
    'age': age,
    'monthly_salary': monthly_salary,
    'current_emi_amount': current_emi_amount,
    'existing_loans': existing_loans,
    'other_monthly_expenses': other_monthly_expenses,
    'credit_score': credit_score,
    'requested_amount': requested_amount,
    'affordability_ratio': affordability_ratio,
    'debt_to_income': debt_to_income
}
df = pd.DataFrame([data])

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def align_features(model, df):
    """Ensure input features match model expectations."""
    if hasattr(model, "feature_names_in_"):
        for feature in model.feature_names_in_:
            if feature not in df.columns:
                df[feature] = 0
        df = df[model.feature_names_in_]
    return df

def extract_shap_vector(shap_values, explainer, df_aligned):
    """Robustly extract a single 1-D SHAP explanation vector."""
    sv = shap_values
    if isinstance(sv, list):
        idx = 1 if len(sv) > 1 else 0
        arr = np.array(sv[idx])
        base = np.array(explainer.expected_value)[idx] if hasattr(explainer, 'expected_value') else None
        if arr.ndim == 2:
            return arr[0], base
        return arr.flatten(), base
    arr = np.array(sv)
    if arr.ndim == 2 and arr.shape[0] == df_aligned.shape[0]:
        return arr[0], getattr(explainer, 'expected_value', None)
    if arr.ndim == 3:
        cls = 1 if arr.shape[-1] > 1 else 0
        return arr[0, :, cls], getattr(explainer, 'expected_value', None)
    return arr.flatten(), getattr(explainer, 'expected_value', None)

def safe_text(txt):
    """Convert emojis and smart quotes to PDF-safe ASCII."""
    replacements = {
        '’': "'", '‘': "'", '“': '"', '”': '"',
        '✅': 'Eligible', '❌': 'Not Eligible', '🎯': 'Result:',
        '💰': 'EMIPredictAI', '👨‍💻': 'Developer', '🏫': 'College'
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)
    return txt.encode('latin-1', 'replace').decode('latin-1')

# =========================================================
# CLASSIFICATION (with SHAP + PDF)
# =========================================================
eligibility_result = None

if st.button("🔍 Predict EMI Eligibility"):
    if clf is None:
        st.error("⚠️ Classification model not found.")
    else:
        df_aligned = align_features(clf, df)
        pred = clf.predict(df_aligned)[0]
        eligibility_result = pred
        result = "✅ Eligible" if pred == 1 else "❌ Not Eligible"
        st.success(f"🎯 Predicted EMI Eligibility: {result}")

        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(df_aligned)
            st.markdown("### 💡 Feature Impact Explanation")

            shap_vec, base_val = extract_shap_vector(shap_values, explainer, df_aligned)
            shap_vec = np.ravel(shap_vec).astype(float)
            if len(shap_vec) != df_aligned.shape[1]:
                shap_vec = np.resize(shap_vec, (df_aligned.shape[1],))
            base_val = float(np.ravel(base_val)[0]) if isinstance(base_val, (list, np.ndarray)) else float(base_val or 0)

            shap_exp = shap.Explanation(
                values=shap_vec,
                base_values=base_val,
                data=df_aligned.iloc[0].values.astype(float),
                feature_names=df_aligned.columns.tolist()
            )

            # --- SHAP Waterfall plot ---
            fig, ax = plt.subplots(figsize=(9, 4))
            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig, clear_figure=True)

            # --- SHAP Bar plot ---
            st.markdown("#### 📊 Overall Feature Importance (bar)")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
            shap.summary_plot(np.array([shap_vec]), df_aligned, plot_type="bar", show=False)
            st.pyplot(fig_bar, clear_figure=True)

            # Save SHAP bar chart
            bar_img_path = os.path.join(base, "assets", "shap_bar.png")
            fig_bar.savefig(bar_img_path, bbox_inches="tight")

            # --- Generate PDF Report ---
            pdf = FPDF()
            pdf.add_page()

            if os.path.exists(logo_path):
                pdf.image(logo_path, x=85, y=10, w=40)
                pdf.ln(30)

            pdf.set_font("Arial", "B", 16)
            pdf.cell(200, 10, txt=safe_text("MGM’s College of Engineering, Nanded"), ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(200, 8, txt=safe_text("Department of Computer Science & Engineering"), ln=True, align="C")
            pdf.cell(200, 8, txt=safe_text("Project: EMIPredictAI"), ln=True, align="C")
            pdf.ln(8)
            pdf.cell(200, 8, txt=safe_text("Developed by: Pruthviraj Tarode (B.Tech CSE)"), ln=True, align="C")
            pdf.ln(12)

            # Applicant details
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Applicant Details", ln=True)
            pdf.set_font("Arial", "", 12)
            for col, val in data.items():
                pdf.cell(200, 8, txt=safe_text(f"{col}: {round(val, 2)}"), ln=True)
            pdf.ln(8)

            # Prediction result
            plain_result = result.replace("✅", "Eligible").replace("❌", "Not Eligible")
            pdf.cell(200, 8, txt=safe_text(f"Prediction Result: {plain_result}"), ln=True)

            # Embed SHAP image
            if os.path.exists(bar_img_path):
                pdf.image(bar_img_path, x=25, w=160)

            pdf.output("EMI_Report.pdf")

            with open("EMI_Report.pdf", "rb") as f:
                st.download_button("📄 Download EMI Report (PDF)", f, file_name="EMIPredictAI_Report.pdf")

        except Exception as e:
            st.warning(f"⚠️ SHAP visualization issue: {e}")

# =========================================================
# REGRESSION SECTION (Smart Eligibility Logic)
# =========================================================
if st.button("💹 Predict Maximum Monthly EMI"):
    if reg is None:
        st.error("⚠️ Regression model not found.")
    else:
        df_aligned = align_features(reg, df)
        if clf is not None:
            pred = clf.predict(df_aligned)[0]
            if pred == 0:
                st.warning("⚠️ Applicant is Not Eligible for EMI estimation.")
            else:
                val = reg.predict(df_aligned)[0]
                st.success(f"💵 Estimated Maximum EMI: ₹{val:,.2f}")
        else:
            val = reg.predict(df_aligned)[0]
            st.success(f"💵 Estimated Maximum EMI: ₹{val:,.2f}")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<hr><p style='text-align:center; color:gray;'>© 2025 Developed by Pruthviraj Tarode — MGM’s College of Engineering, Nanded</p>",
    unsafe_allow_html=True
)
