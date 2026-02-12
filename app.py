import streamlit as st
import os
from groq import Groq
from prediction.predict_hybrid import predict_academic_progression

# Set page config
st.set_page_config(
    page_title="Academic Progression System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .recommend-btn > button {
        background-color: #4CAF50;
        color: white;
    }
    .predict-btn > button {
        background-color: #2196F3;
        color: white;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_recommendations(api_key, student_data):
    if not api_key:
        return "⚠️ Please provide a valid Groq API Key in the sidebar to get AI-powered recommendations."
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        Act as an expert academic counselor. Based on the following student profile, provide specific, actionable, and encouraging recommendations to improve their academic performance and study habits.
        
        Student Profile:
        - Gender: {student_data['Gender']}
        - Study Hours per Day: {student_data['Study Hours']}
        - Attendance: {student_data['Attendance']}%
        - Semester Grades (Aggregates): s1:{student_data['Sem 1']}, s2:{student_data['Sem 2']}, s3:{student_data['Sem 3']}, s4:{student_data['Sem 4']}, s5:{student_data['Sem 5']}, s6:{student_data['Sem 6']}, s7:{student_data['Sem 7']}, s8:{student_data['Sem 8']}
        - Total Aggregate: {student_data['Total Aggregate']}
        - Backlogs: {student_data['Backlogs']}
        - Internet Access: {student_data['Internet Access']}
        - Extracurricular Activities: {student_data['Extracurriculars']}
        - Library Usage: {student_data['Library Usage']}
        - Sleep Hours: {student_data['Sleep Hours']}
        - Classroom Engagement: {student_data['Classroom Engagement']}
        - Peer Interaction: {student_data['Peer Interaction']}
        - Devices Used for Studying: {', '.join(student_data['Devices Used'])}
        
        Please analyze this profile and provide a SHORT, CONCISE, and SIMPLIFIED response. Do NOT use long paragraphs. Use bullet points.
        
        Focus ONLY on these three things:
        1. **Weaknesses**: What is negatively impacting their performance? (Max 2-3 points)
        2. **Areas of Improvement**: Specific metrics or habits to target. (Max 2-3 points)
        3. **How to Overcome**: Simple, actionable steps to fix the issues. (Max 3 steps)
        
        Keep the language simple and direct.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful and encouraging academic counselor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Sidebar for Settings
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/student-male.png", width=100)
    st.title("Settings")
    api_key = st.text_input("Enter Groq API Key", type="password", help="Get your key from Groq Console")
    st.write("---")
    st.info("This tool provides both AI-powered study recommendations and academic risk prediction.")

# Main Title
st.title("🎓 Academic Progression System")
st.write("Enter your details below to get recommendations or predict academic risk.")

# Form Input
with st.form("student_form"):
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("👤 Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        sleep_hours = st.selectbox("Average Sleep Hours", ["<2 hrs", "2-4 hrs", "4-6 hrs", "6-8 hrs", ">8 hrs"], index=3)
        internet_access = st.radio("Internet Access", ["Yes", "No"], horizontal=True)
        extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

    with col2:
        st.header("📚 Study Habits")
        study_hours = st.number_input("Study Hours (Daily)", min_value=0.0, max_value=24.0, step=0.5, value=4.0)
        attendance = st.slider("Attendance Percentage", 0, 100, 75)
        library_usage = st.selectbox("Library Usage", ["Yes", "No"])
        devices = st.multiselect("Devices Used for Studying", ["Laptops", "Mobile", "Text books"], default=["Text books"])

    with col3:
        st.header("🏫 Engagement")
        # Default keys for engagement
        engagement_map = {"Low": 0, "Medium": 1, "High": 2} # Just for select slider logic if needed, but strings are fine
        classroom_engagement = st.select_slider("Classroom Engagement Level", options=["Low", "Medium", "High"], value="Medium")
        peer_interaction = st.select_slider("Peer Interaction Level", options=["Low", "Medium", "High"], value="Medium")
        backlogs = st.number_input("Number of Backlogs", min_value=0, step=1, value=0)

    st.write("---")
    st.header("📊 Academic Performance (Semester Aggregates)")
    
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        sem1 = st.number_input("Sem 1 %", 0.0, 100.0, 60.0)
        sem5 = st.number_input("Sem 5 %", 0.0, 100.0, 0.0)
    with scol2:
        sem2 = st.number_input("Sem 2 %", 0.0, 100.0, 60.0)
        sem6 = st.number_input("Sem 6 %", 0.0, 100.0, 0.0)
    with scol3:
        sem3 = st.number_input("Sem 3 %", 0.0, 100.0, 60.0)
        sem7 = st.number_input("Sem 7 %", 0.0, 100.0, 0.0)
    with scol4:
        sem4 = st.number_input("Sem 4 %", 0.0, 100.0, 60.0)
        sem8 = st.number_input("Sem 8 %", 0.0, 100.0, 0.0)
        
    total_aggregate = st.number_input("Total Aggregate %", 0.0, 100.0, 60.0)

    # Buttons
    st.write("---")
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        submit_recommend = st.form_submit_button("✨ Give Recommendations")
    with b_col2:
        submit_predict = st.form_submit_button("🔮 Predict Risk Status")

# common_data preparation
student_data = {
    "Gender": gender, "Sleep Hours": sleep_hours, "Internet Access": internet_access,
    "Extracurriculars": extracurricular, "Study Hours": study_hours, "Attendance": attendance,
    "Library Usage": library_usage, "Devices Used": devices, "Classroom Engagement": classroom_engagement,
    "Peer Interaction": peer_interaction, "Backlogs": backlogs,
    "Sem 1": sem1, "Sem 2": sem2, "Sem 3": sem3, "Sem 4": sem4,
    "Sem 5": sem5, "Sem 6": sem6, "Sem 7": sem7, "Sem 8": sem8,
    "Total Aggregate": total_aggregate,
    
    # Map for Prediction inputs specifically (keys matching predict_hybrid expectations if different)
    'Semester 1(aggregate)': sem1,
    'Semester 2(aggregate)': sem2,
    'Semester 3(aggregate)': sem3,
    'Semester 4(aggregate)': sem4
}

# --- Recommendations Logic ---
if submit_recommend:
    st.write("---")
    st.header("📋 AI Recommendations")
    with st.spinner("AI is analyzing your profile..."):
        recommendation = get_recommendations(api_key, student_data)
        st.markdown(recommendation)

# --- Prediction Logic ---
if submit_predict:
    st.write("---")
    st.header("📉 Risk Prediction Analysis")
    with st.spinner("Analyzing performance patterns..."):
        # Pass the full student_data dict. predict_hybrid will look for its specific keys.
        label, confidence = predict_academic_progression(student_data)
        
    # Result Display
    status = label[0]
    conf_score = confidence[0] * 100
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        if status == "At Risk":
            st.error(f"### ⚠️ Status: At Risk")
        else:
            st.success(f"### ✅ Status: At Safe")
        st.metric("Confidence Score", f"{conf_score:.2f}%")

    with col_res2:
        if status == "At Risk":
            st.write("### Analysis")
            st.warning("Based on your academic history and study habits, there is a high probability of academic decline. We recommend focusing on inconsistent areas and seeking guidance.")
        else:
            st.write("### Analysis")
            st.success("You are on a safe academic path! Your performance and habits indicate stability. Keep up the consistent effort.")
            
    # Chart
    st.write("#### Grade Trend (Sem 1-4)")
    chart_data = {
        "Semester": ["Sem 1", "Sem 2", "Sem 3", "Sem 4"],
        "Aggregate": [sem1, sem2, sem3, sem4]
    }
    st.line_chart(chart_data, x="Semester", y="Aggregate", color="#FF0000" if status == "At Risk" else "#00FF00")
