import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# 1. Load the trained model, scaler, and encoder
# -------------------------
model = joblib.load("models/super_model.pkl")
scaler = joblib.load("models/super_scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# -------------------------
# 2. App Config
# -------------------------
st.set_page_config(page_title="TaskWise – ML Prioritizer", layout="centered")
st.title("📊 TaskWise – ML Task Prioritizer")
st.markdown(
    "Enter task details to get a smart priority prediction powered by a trained machine learning model."
)

# -------------------------
# 3. Form Inputs
# -------------------------
with st.form("task_form"):
    task_name = st.text_input("📝 Task Name", placeholder="e.g., Complete project report")
    importance = st.slider("🔴 Importance (1 = Low, 5 = High)", 1, 5, 3)
    effort = st.slider("⏱ Effort (Hours Required)", 1, 10, 5)
    deadline = st.date_input("📅 Deadline")
    submitted = st.form_submit_button("🚀 Predict Priority")

# -------------------------
# 4. Prediction Logic
# -------------------------
if submitted and task_name and deadline:
    try:
        # Convert deadline to Timestamp
        deadline_dt = pd.to_datetime(deadline)
        today = pd.Timestamp.today().normalize()
        days_left = (deadline_dt - today).days

        if days_left < 0:
            st.warning("⚠️ Deadline is in the past. Please select a future date.")
        else:
            # Prepare input features
            features = pd.DataFrame([[importance, effort, days_left]],
                                    columns=["Importance", "Effort", "Days_Left"])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            priority_label = label_encoder.inverse_transform(prediction)[0]

            # Show result
            st.subheader("✅ Prediction Result")
            st.markdown(f"**Task:** {task_name}")
            st.markdown(f"**Predicted Priority:** :blue[{priority_label}]")

            result_df = pd.DataFrame({
                "Task Name": [task_name],
                "Importance": [importance],
                "Effort": [effort],
                "Days Left": [days_left],
                "Predicted Priority": [priority_label]
            })
            st.table(result_df)

            # Store prediction history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append(result_df.iloc[0].to_dict())

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------------
# 5. History Table + Download
# -------------------------
if "history" in st.session_state and st.session_state.history:
    st.subheader("📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="taskwise_predictions.csv",
        mime="text/csv"
    )
