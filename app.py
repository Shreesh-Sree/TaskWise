import streamlit as st
import pandas as pd
import joblib
from datetime import date
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model/super_model.pkl")
scaler = joblib.load("model/super_scaler.pkl")
priority_map = {0: "Low", 1: "Medium", 2: "High"}

# Initialize task storage
if "tasks" not in st.session_state:
    st.session_state.tasks = []

st.set_page_config(page_title="TaskWise", layout="centered")
st.title("🧠 TaskWise – ML Task Prioritizer")

st.markdown("Enter task details. Let ML predict what’s most important!")

# Input form
with st.form("task_form"):
    name = st.text_input("Task Name")
    importance = st.slider("Importance (1-5)", 1, 5, 3)
    effort = st.slider("Effort (in hours)", 1, 10, 2)
    deadline = st.date_input("Deadline", min_value=date.today())
    submit = st.form_submit_button("Add Task")

if submit:
    days_left = (deadline - date.today()).days
    scaled = scaler.transform([[importance, effort, days_left]])
    predicted = model.predict(scaled)[0]
    priority = priority_map[int(predicted)]

    st.session_state.tasks.append({
        "Task Name": name,
        "Importance": importance,
        "Effort": effort,
        "Deadline": deadline.strftime("%Y-%m-%d"),
        "Days Left": days_left,
        "Predicted Priority": priority
    })
    st.success(f"✅ '{name}' added as **{priority} Priority**")

# Display Table
if st.session_state.tasks:
    df = pd.DataFrame(st.session_state.tasks)
    st.subheader("📋 Prioritized Task List")
    st.dataframe(df.style.applymap(
        lambda x: "color:red;" if x == "High" else
                  "color:orange;" if x == "Medium" else
                  "color:green;",
        subset=["Predicted Priority"]
    ))

    # Visualization
    st.subheader("📊 Priority Distribution")
    pie_data = df["Predicted Priority"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

    # Export
    st.download_button("📁 Download as CSV", df.to_csv(index=False), file_name="taskwise_tasks.csv", mime="text/csv")
