import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure the models are trained and saved.")
        st.stop()

# Load data for reference
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('students_dropout_academic_success.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file not found.")
        return None

model, scaler, label_encoder = load_models()
df_original = load_data()

# Title and Header
st.title("🎓 Student Dropout Prediction System")
st.markdown("### Predicting Student Success and Dropout Risk")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Predictions", "📈 Data Analysis", "ℹ️ About Model"]
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome! 👋
        
        This application uses a **CatBoost Machine Learning Model** to predict student dropout status based on academic and demographic features.
        
        ### Key Features:
        - 📊 **Real-time Predictions** - Get instant dropout risk assessments
        - 📈 **Data Analysis** - Explore patterns in student data
        - 🎯 **Model Information** - Learn about the model's performance
        - ✨ **User-friendly Interface** - Simple and intuitive design
        
        ### How to Use:
        1. Navigate to **Predictions** page
        2. Enter student information
        3. Click **Predict** to get results
        4. View risk assessment and recommendations
        """)
    
    with col2:
        st.metric("Model Type", "CatBoost Classifier")
        st.metric("Target Classes", len(label_encoder.classes_))
        st.metric("Training Samples", len(df_original) if df_original is not None else "N/A")

# ==================== PREDICTIONS PAGE ====================
elif page == "📊 Predictions":
    st.header("Make Predictions")
    st.markdown("Enter student information to predict dropout risk")
    st.markdown("---")
    
    if df_original is not None:
        # Get feature names and ranges
        numeric_features = df_original.select_dtypes(include=np.number).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        # Create input form
        st.subheader("📋 Student Information")
        
        col1, col2, col3 = st.columns(3)
        input_data = {}
        
        with st.form("prediction_form"):
            # Create inputs for each feature
            for idx, feature in enumerate(numeric_features):
                col = [col1, col2, col3][idx % 3]
                
                feature_min = float(df_original[feature].min())
                feature_max = float(df_original[feature].max())
                feature_mean = float(df_original[feature].mean())
                
                with col:
                    input_data[feature] = st.slider(
                        f"{feature}",
                        min_value=feature_min,
                        max_value=feature_max,
                        value=feature_mean,
                        step=(feature_max - feature_min) / 100
                    )
            
            # Prediction button
            st.markdown("---")
            predict_button = st.form_submit_button(
                "🎯 Make Prediction",
                use_container_width=True
            )
        
        if predict_button:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure features are in the same order as training
            input_df = input_df[numeric_features]
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Get prediction label
            predicted_class = label_encoder.classes_[prediction]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Predicted Status")
                if "Dropout" in predicted_class or "Dropout" == predicted_class:
                    st.error(f"### {predicted_class}")
                elif "Enrolled" in predicted_class or "Enrolled" == predicted_class:
                    st.success(f"### {predicted_class}")
                else:
                    st.info(f"### {predicted_class}")
            
            with col2:
                st.markdown("### Confidence")
                max_prob = max(prediction_proba)
                st.metric("Confidence Score", f"{max_prob*100:.2f}%")
            
            with col3:
                st.markdown("### Risk Level")
                if max_prob < 0.6:
                    st.warning("⚠️ LOW CONFIDENCE")
                elif max_prob < 0.8:
                    st.info("🟡 MEDIUM CONFIDENCE")
                else:
                    st.success("✅ HIGH CONFIDENCE")
            
            # Show probability distribution
            st.markdown("---")
            st.subheader("📈 Probability Distribution")
            
            prob_data = {
                'Status': label_encoder.classes_,
                'Probability': prediction_proba
            }
            prob_df = pd.DataFrame(prob_data)
            
            fig = px.bar(
                prob_df,
                x='Status',
                y='Probability',
                color='Status',
                height=400,
                title="Prediction Probabilities for Each Class"
            )
            fig.update_layout(showlegend=False, yaxis_title="Probability", xaxis_title="Status")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if "Dropout" in predicted_class or "Dropout" == predicted_class:
                st.warning("""
                ### High Dropout Risk Detected
                - **Schedule Academic Counseling** - Meet with academic advisors
                - **Peer Mentoring** - Pair with successful students
                - **Study Groups** - Encourage collaboration and support
                - **Financial Support** - Check for financial aid programs
                - **Mental Health Resources** - Connect with counseling services
                """)
            else:
                st.success("""
                ### Low Dropout Risk
                - **Continue Current Path** - Student is on a positive trajectory
                - **Advanced Opportunities** - Consider honors or advanced courses
                - **Peer Mentoring** - Could help other struggling students
                - **Career Planning** - Begin thinking about post-graduation plans
                """)

# ==================== DATA ANALYSIS PAGE ====================
elif page == "📈 Data Analysis":
    st.header("Data Analysis & Insights")
    st.markdown("---")
    
    if df_original is not None:
        # Display dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df_original))
        with col2:
            st.metric("Total Features", df_original.shape[1])
        with col3:
            st.metric("Missing Values", df_original.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df_original.duplicated().sum())
        
        st.markdown("---")
        
        # Tab selection for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Statistics", "Features"])
        
        with tab1:
            st.subheader("Feature Distributions")
            numeric_features = df_original.select_dtypes(include=np.number).columns.tolist()
            if 'target' in numeric_features:
                numeric_features.remove('target')
            
            selected_feature = st.selectbox("Select Feature", numeric_features)
            
            fig = px.histogram(
                df_original,
                x=selected_feature,
                nbins=50,
                title=f"Distribution of {selected_feature}",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Correlation Analysis")
            numeric_df = df_original.select_dtypes(include=np.number)
            correlation = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(height=600, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Statistical Summary")
            st.dataframe(df_original.describe(), use_container_width=True)
        
        with tab4:
            st.subheader("Feature Information")
            feature_info = pd.DataFrame({
                'Feature': df_original.columns,
                'Type': df_original.dtypes,
                'Non-Null Count': df_original.notnull().sum(),
                'Null Count': df_original.isnull().sum()
            })
            st.dataframe(feature_info, use_container_width=True)

# ==================== ABOUT MODEL PAGE ====================
elif page == "ℹ️ About Model":
    st.header("Model Information & Performance")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Details")
        st.markdown("""
        **Model Type:** CatBoost Classifier
        
        **Purpose:** Predict student dropout status
        
        **Training Approach:**
        - Train/Test Split: 80/20 (stratified)
        - Class Balancing: SMOTE
        - Feature Scaling: StandardScaler
        - Hyperparameters:
          - Iterations: 500
          - Learning Rate: 0.05
          - Depth: 8
          - Loss Function: MultiClass
        """)
    
    with col2:
        st.subheader("🎯 Model Performance")
        st.markdown("""
        **Models Tested:**
        - Random Forest
        - Extra Trees
        - Gradient Boosting
        - XGBoost
        - LightGBM
        - **CatBoost ✓** (Selected)
        
        **Why CatBoost?**
        - Best accuracy score
        - Handles categorical data well
        - Robust to overfitting
        - Fast training and prediction
        """)
    
    st.markdown("---")
    st.subheader("📚 Data Preprocessing Steps")
    
    steps = [
        "1️⃣ **Data Loading** - Load student dataset",
        "2️⃣ **Duplicate Removal** - Remove duplicate records",
        "3️⃣ **Missing Value Imputation** - Fill with median values",
        "4️⃣ **Feature Encoding** - Encode target variable",
        "5️⃣ **Data Splitting** - Train (80%) / Test (20%)",
        "6️⃣ **Feature Scaling** - StandardScaler normalization",
        "7️⃣ **Class Balancing** - SMOTE for imbalanced classes",
        "8️⃣ **Model Training** - Train CatBoost classifier",
        "9️⃣ **Model Evaluation** - Accuracy, F1-Score, Classification Report"
    ]
    
    for step in steps:
        st.markdown(f"**{step}**")
    
    st.markdown("---")
    st.subheader("🔑 Key Features Used")
    
    if df_original is not None:
        numeric_features = df_original.select_dtypes(include=np.number).columns.tolist()
        if 'target' in numeric_features:
            numeric_features.remove('target')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Total Features:** {len(numeric_features)}")
        with col2:
            st.info(f"**Target Variable:** Student Status")
        with col3:
            st.info(f"**Classes:** {', '.join(label_encoder.classes_)}")
        
        st.markdown("**Feature List:**")
        features_text = ", ".join(numeric_features[:5]) + "..."
        st.text(features_text)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Student Dropout Prediction System | Built with Streamlit & CatBoost</p>
    <p style='font-size: 0.8em'>© 2024 | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
