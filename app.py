import streamlit as st
import pandas as pd
import os
import joblib

# -------------------------
# 1. Load the trained model, scaler, and label encoder
# -------------------------
model_path = os.path.join("models", "super_model.pkl")
scaler_path = os.path.join("models", "super_scaler.pkl")
encoder_path = os.path.join("models", "label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# -------------------------
# 2. Streamlit App Configuration
# -------------------------
st.set_page_config(page_title="TaskWise ML Prioritizer", layout="centered")

st.title("TaskWise Machine Learning Task Prioritizer")
st.write(
    """
    Enter your task details below to get an intelligent priority prediction
    powered by a Random Forest Classifier.
    """
)

# -------------------------
# 3. Input Form
# -------------------------
with st.form("task_form"):
    task_name = st.text_input("Task Name", placeholder="e.g., Finish Report")
    importance = st.slider("Importance (1 = Low, 5 = High)", 1, 5, 3)
    effort = st.slider("Effort Required (hours)", 1, 10, 5)
    deadline = st.date_input("Deadline (select date)")
    submitted = st.form_submit_button("Predict Priority")

# -------------------------
# 4. Prediction Logic
# -------------------------
if submitted and task_name:
    # Calculate days left
    days_left = (deadline - pd.Timestamp.today().normalize()).days
    if days_left < 0:
        st.warning("The selected deadline is in the past. Please choose a future date.")
    else:
        # Prepare features
        features = pd.DataFrame([[importance, effort, days_left]],
                                 columns=["Importance", "Effort", "Days_Left"])
        # Scale features
        features_scaled = scaler.transform(features)
        # Predict and decode label
        pred_encoded = model.predict(features_scaled)
        priority_label = label_encoder.inverse_transform(pred_encoded)[0]
        
        # Display results
        st.subheader("Prediction Result")
        st.markdown(f"**Task:** {task_name}")
        st.markdown(f"**Predicted Priority:** :blue[{priority_label}]")
        
        # Optional: Show DataFrame with result
        result_df = pd.DataFrame({
            "Task Name": [task_name],
            "Importance": [importance],
            "Effort": [effort],
            "Days Left": [days_left],
            "Predicted Priority": [priority_label]
        })
        st.table(result_df)
        
        # Pie chart for illustration (static, with only this task)
        st.subheader("Priority Distribution (Current Task)")
        chart_data = pd.DataFrame({
            'Priority': [priority_label],
            'Count': [1]
        }).set_index('Priority')
        st.bar_chart(chart_data)

# -------------------------
# 5. Optional: Download History
# -------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

if submitted and task_name and days_left >= 0:
    st.session_state.history.append({
        "Task Name": task_name,
        "Importance": importance,
        "Effort": effort,
        "Days Left": days_left,
        "Predicted Priority": priority_label
    })

if st.session_state.history:
    st.subheader("All Predicted Tasks")
    st.dataframe(pd.DataFrame(st.session_state.history))
    csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV of Predictions",
        data=csv,
        file_name='predicted_tasks.csv',
        mime='text/csv'
    )
