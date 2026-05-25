# STUDENT PERFORMANCE DASHBOARD
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# PAGE CONFIG

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.big-title {
    font-size: 45px;
    font-weight: 800;
    color: #1f3b73;
}

.subtitle {
    font-size: 20px;
    color: #555;
}

.metric-card {
    background: linear-gradient(135deg, #1f77b4, #4fa3ff);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
}

.section-title {
    font-size: 30px;
    font-weight: bold;
    margin-top: 20px;
    color: #1f3b73;
}

.recommend-box {
    background-color: #ffffff;
    padding: 20px;
    border-left: 6px solid #1f77b4;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# LOAD DATA
df = pd.read_csv("xAPI-Edu-Data.csv")

# ENCODE CATEGORICAL COLUMNS
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# SPLIT FEATURES & TARGET
X = df.drop("Class", axis=1)
y = df["Class"]

# TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

# SIDEBAR
st.sidebar.title("🎛 Student Input Panel")

gender = st.sidebar.selectbox(
    "Gender",
    [0,1],
    help="0 = Female, 1 = Male"
)

Nationality = st.sidebar.slider(
    "Nationality",
    0,13,4,
    help="Encoded nationality category"
)

PlaceofBirth = st.sidebar.slider(
    "Place of Birth",
    0,13,4
)

StageID = st.sidebar.slider(
    "Stage ID",
    0,2,1
)

GradeID = st.sidebar.slider(
    "Grade ID",
    0,9,4
)

SectionID = st.sidebar.slider(
    "Section ID",
    0,2,0
)

Topic = st.sidebar.slider(
    "Topic",
    0,11,5
)

Semester = st.sidebar.selectbox(
    "Semester",
    [0,1]
)

Relation = st.sidebar.selectbox(
    "Relation",
    [0,1]
)

raisedhands = st.sidebar.slider(
    "Raised Hands",
    0,100,50,
    help="Class participation level"
)

VisITedResources = st.sidebar.slider(
    "Visited Resources",
    0,100,60,
    help="Educational resource engagement"
)

AnnouncementsView = st.sidebar.slider(
    "Announcements View",
    0,100,40
)

Discussion = st.sidebar.slider(
    "Discussion",
    0,100,40
)

ParentAnsweringSurvey = st.sidebar.selectbox(
    "Parent Answering Survey",
    [0,1]
)

ParentschoolSatisfaction = st.sidebar.selectbox(
    "Parent School Satisfaction",
    [0,1]
)

StudentAbsenceDays = st.sidebar.selectbox(
    "Student Absence Days",
    [0,1]
)

# USER DATAFRAME
input_data = pd.DataFrame({
    'gender':[gender],
    'NationalITy':[Nationality],
    'PlaceofBirth':[PlaceofBirth],
    'StageID':[StageID],
    'GradeID':[GradeID],
    'SectionID':[SectionID],
    'Topic':[Topic],
    'Semester':[Semester],
    'Relation':[Relation],
    'raisedhands':[raisedhands],
    'VisITedResources':[VisITedResources],
    'AnnouncementsView':[AnnouncementsView],
    'Discussion':[Discussion],
    'ParentAnsweringSurvey':[ParentAnsweringSurvey],
    'ParentschoolSatisfaction':[ParentschoolSatisfaction],
    'StudentAbsenceDays':[StudentAbsenceDays]
})

# PREDICTION
prediction = model.predict(input_data)[0]

# TABS
tab1, tab2, tab3 = st.tabs([
    "🎯 Prediction Dashboard",
    "📊 Dataset Analytics",
    "ℹ About Project"
])

# TAB 1

with tab1:

    st.markdown("""
    <div class='big-title'>
    🎓 AI-Powered Student Performance Analytics System
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='subtitle'>
    Predicting academic performance using Machine Learning and behavioral analytics.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
        <h2>82.29%</h2>
        <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
        <h2>478</h2>
        <p>Total Students</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
        <h2>16</h2>
        <p>Features Used</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
        <h2>Random Forest</h2>
        <p>Best Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # PREDICTION RESULT

    st.markdown("<div class='section-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)

    if prediction == 2:

        st.success("🏆 High Academic Performance Predicted")

        st.markdown("""
        <div class='recommend-box'>

        ### 📈 Interpretation
        This student demonstrates strong academic engagement and learning behavior.

        ### ✅ Key Positive Indicators
        - High classroom participation
        - Strong learning resource usage
        - Active discussions
        - Better attendance pattern

        ### 💡 Recommendation
        Continue maintaining consistent learning habits and participation.

        </div>
        """, unsafe_allow_html=True)

    elif prediction == 1:

        st.warning("⚡ Medium Academic Performance Predicted")

        st.markdown("""
        <div class='recommend-box'>

        ### 📈 Interpretation
        The student is performing moderately but has potential for improvement.

        ### 🔍 Areas to Improve
        - Increase discussion participation
        - Improve attendance
        - Use more learning resources

        ### 💡 Recommendation
        Additional mentoring and consistent study routines are suggested.

        </div>
        """, unsafe_allow_html=True)

    else:

        st.error("⚠ Low Academic Performance Predicted")

        st.markdown("""
        <div class='recommend-box'>

        ### 📈 Interpretation
        Student may require academic support and monitoring.

        ### 🚨 Risk Indicators
        - Low engagement
        - Poor attendance
        - Limited classroom interaction

        ### 💡 Recommendation
        Immediate academic intervention and parental guidance are recommended.

        </div>
        """, unsafe_allow_html=True)

    # FEATURE IMPORTANCE

    st.markdown("<div class='section-title'>📊 Feature Importance</div>", unsafe_allow_html=True)

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance,
        ax=ax
    )
    st.pyplot(fig)

    # STUDENT DATA

    st.markdown("<div class='section-title'>📋 Student Input Data</div>", unsafe_allow_html=True)
    st.dataframe(input_data)

# TAB 2

with tab2:

    st.markdown("<div class='section-title'>📊 Dataset Insights</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        fig1, ax1 = plt.subplots(figsize=(5,5))

        performance_counts = df['Class'].value_counts()

        ax1.pie(
            performance_counts,
            labels=["High","Medium","Low"],
            autopct='%1.1f%%'
        )

        ax1.set_title("Performance Distribution")
        st.pyplot(fig1)

    with col2:

        fig2, ax2 = plt.subplots(figsize=(6,4))

        sns.countplot(
            x='gender',
            data=df,
            ax=ax2
        )

        ax2.set_title("Gender Distribution")
        st.pyplot(fig2)

    st.write("")

    # CORRELATION HEATMAP

    st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(12,8))

    sns.heatmap(
    df.corr(numeric_only=True),
        cmap='coolwarm',
        ax=ax3
    )
    st.pyplot(fig3)

# TAB 3

with tab3:

    st.markdown("<div class='section-title'>ℹ About The Project</div>", unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Objective

    This project predicts student academic performance using Machine Learning algorithms.

    ### 🤖 Models Used
    - Logistic Regression
    - Decision Tree
    - Random Forest

    ### 🏆 Best Performing Model
    Random Forest achieved the highest accuracy:
    - Accuracy: 82.29%

    ### 🌍 Real World Applications
    - Early identification of struggling students
    - Academic intervention planning
    - Student engagement monitoring
    - Educational analytics

    ### 🛠 Technologies Used
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - NumPy
    - Seaborn
    - Matplotlib

    ### 👩‍💻 Developed By
    Alvira Parveen  

    """)

    st.success("🚀 AI + Education = Smarter Learning")