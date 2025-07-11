import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Client Retention Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .prediction-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .stTab {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        tfidf_title = joblib.load("artifacts/tfidf_title.pkl")
        title_vec = tfidf_title.transform([data['job_title']])
        tfidf_desc = joblib.load("artifacts/tfidf_desc.pkl")
        le_country = joblib.load("artifacts/le_country.pkl")
        le_state = joblib.load("artifacts/le_state.pkl")
        mlb = joblib.load("artifacts/mlb_tags.pkl")
        model = joblib.load("artifacts/model.pkl")
        return model, tfidf_title, tfidf_desc, le_country, le_state, mlb
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure all model files are compatible and generated from the same training session.")
        st.stop()

@st.cache_data
def get_exchange_rate(from_currency, to_currency='USD'):
    """Get exchange rate from Fixer.io API"""
    if from_currency == 'USD':
        return 1.0
    
    try:
        response = requests.get(f"https://api.exchangerate-api.com/v4/latest/{from_currency}")
        data = response.json()
        return data['rates'].get(to_currency, 1.0)
    except:
        rates = {
            'EUR': 1.1, 'GBP': 1.3, 'CAD': 0.75, 'AUD': 0.7,
            'JPY': 0.0067, 'INR': 0.012, 'BRL': 0.18, 'RUB': 0.011
        }
        return rates.get(from_currency, 1.0)

def preprocess_input(data, tfidf_title, tfidf_desc, le_country, le_state, mlb_tags):
    
    if data['currency'] != 'USD':
        rate = get_exchange_rate(data['currency'])
        data['min_price_usd'] = float(data['min_price'] * rate)
        data['max_price_usd'] = float(data['max_price'] * rate)
        data['avg_price_usd'] = float(data['avg_price'] * rate)
    else:
        data['min_price_usd'] = float(data['min_price'])
        data['max_price_usd'] = float(data['max_price'])
        data['avg_price_usd'] = float(data['avg_price'])
    
    try:
        data['client_country_encoded'] = int(le_country.transform([data['client_country']])[0])
    except:
        data['client_country_encoded'] = 0  
    
    try:
        data['client_state_encoded'] = int(le_state.transform([data['client_state']])[0])
    except:
        data['client_state_encoded'] = 0  
    
    currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'INR', 'NGN', 'ZAR', 'JPY']
    rate_types = ['Fixed', 'Hourly']
    currency_features = []
    for currency in currencies:
        currency_features.append(1 if data['currency'] == currency else 0)
    rate_type_features = []
    for rate_type in rate_types:
        rate_type_features.append(1 if data['rate_type'] == rate_type else 0)
    
    try:
        title_vec = tfidf_title.transform([data['job_title']]).toarray().flatten()
        desc_vec = tfidf_desc.transform([data['job_description']]).toarray().flatten()
        expected_title_len = tfidf_title.idf_.shape[0]
        if title_vec.shape[0] < expected_title_len:
            padding = np.zeros(expected_title_len - title_vec.shape[0])
            title_vec = np.concatenate([title_vec, padding])
        elif title_vec.shape[0] > expected_title_len:
            title_vec = title_vec[:expected_title_len]
        expected_desc_len = tfidf_desc.idf_.shape[0]
        if desc_vec.shape[0] < expected_desc_len:
            padding = np.zeros(expected_desc_len - desc_vec.shape[0])
            desc_vec = np.concatenate([desc_vec, padding])
        elif desc_vec.shape[0] > expected_desc_len:
            desc_vec = desc_vec[:expected_desc_len]
    except Exception as e:
        st.error(f"Error processing text features: {e}")
        return None
    try:
        tags_list = [tag.strip().lower() for tag in data['tags'].split(',') if tag.strip()]
        tags_vec = mlb_tags.transform([tags_list]).toarray().flatten()

        expected_tag_len = mlb_tags.classes_.shape[0]
        if tags_vec.shape[0] < expected_tag_len:
            padding = np.zeros(expected_tag_len - tags_vec.shape[0])
            tags_vec = np.concatenate([tags_vec, padding])
        elif tags_vec.shape[0] > expected_tag_len:
            tags_vec = tags_vec[:expected_tag_len]
    except Exception as e:
        st.error(f"Error processing tags: {e}")
        return None
    numerical_features = [
        float(data['review_count']), 
        float(data['avg_rating']), 
        float(data['min_price_usd']),
        float(data['max_price_usd']), 
        float(data['avg_price_usd']), 
        float(data['client_country_encoded']),
        float(data['client_state_encoded'])
    ]
    
    try:
        feature_vector = np.concatenate([
            np.array(numerical_features, dtype=np.float64),
            np.array(currency_features, dtype=np.float64),
            np.array(rate_type_features, dtype=np.float64),
            title_vec.astype(np.float64),
            desc_vec.astype(np.float64),
            tags_vec.astype(np.float64)
        ])
        return feature_vector.reshape(1, -1)
    except Exception as e:
        st.error(f"Error creating feature vector: {e}")
        return None
def main():
    st.markdown('<h1 class="main-header">📊 Client Retention Predictor</h1>', unsafe_allow_html=True)
    model, tfidf_title, tfidf_desc, le_country, le_state, mlb_tags = load_models()    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Analytics", "Model Info"])
    
    if page == "Prediction":
        prediction_page(model, tfidf_title, tfidf_desc, le_country, le_state, mlb_tags)
    elif page == "Analytics":
        analytics_page()
    else:
        model_info_page()

def prediction_page(model, tfidf_title, tfidf_desc, le_country, le_state, mlb_tags):
    st.header("🎯 Predict Client Retention")    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Project Details")
        job_title = st.text_input("Job Title", placeholder="e.g., Full Stack Web Developer")
        job_description = st.text_area("Job Description", 
                                     placeholder="Describe the project requirements...",
                                     height=100)
        tags = st.text_input("Project Tags (comma-separated)", 
                           placeholder="e.g., python, web-development, react")
        rate_type = st.selectbox("Rate Type", 
                               ["Fixed", "Hourly"])
        currency = st.selectbox("Currency", 
                              ["USD", "EUR", "GBP", "CAD", "AUD", "INR", "ZAR", "NGN", "JPY"])
    with col2:
        st.subheader("💰 Pricing & Performance")
        min_price = st.number_input("Minimum Price", min_value=0.0, value=100.0)
        max_price = st.number_input("Maximum Price", min_value=0.0, value=500.0)
        avg_price = st.number_input("Average Price", min_value=0.0, value=300.0)
        avg_rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1)
        review_count = st.number_input("Review Count", min_value=0, value=25)
        st.subheader("📍 Client Information")
        client_country = st.selectbox("Client Country", 
                                    ["United States", "United Kingdom", "Canada", 
                                     "Australia", "Germany", "France", "India", 
                                     "Brazil", "Other"])
        client_state = st.text_input("Client State/Province", 
                                   placeholder="e.g., California, Ontario")
    # Prediction button
    if st.button("🚀 Predict Client Retention", type="primary"):
        if not all([job_title, job_description, tags]):
            st.error("Please fill in all required fields (Job Title, Description, Tags)")
            return
        input_data = {
            'job_title': job_title,
            'job_description': job_description,
            'tags': tags,
            'rate_type': rate_type,
            'currency': currency,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'avg_rating': avg_rating,
            'review_count': review_count,
            'client_country': client_country,
            'client_state': client_state if client_state else "Unknown"
        }
        
        try:
            processed_data = preprocess_input(input_data, tfidf_title, tfidf_desc, 
                                            le_country, le_state, mlb_tags)
            if processed_data is None:
                st.error("Error in preprocessing. Please check your input data.")
                return
            # Debug information
            st.info(f"Feature vector shape: {processed_data.shape}")
            st.info(f"Feature vector dtype: {processed_data.dtype}")           
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                retention_prob = probability[1] * 100
                st.metric("Retention Probability", f"{retention_prob:.1f}%")
            with col2:
                risk_level = "Low" if retention_prob > 70 else "Medium" if retention_prob > 50 else "High"
                st.metric("Risk Level", risk_level)
            with col3:
                recommendation = "Retain" if prediction == 1 else "At Risk"
                st.metric("Recommendation", recommendation)
            # Detailed prediction display
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-high">
                    <h3>✅ High Retention Likelihood</h3>
                    <p>This freelancer has a <strong>{retention_prob:.1f}%</strong> probability of retaining the client.</p>
                    <p><strong>Recommendation:</strong> Excellent candidate for client retention.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-low">
                    <h3>⚠️ Low Retention Likelihood</h3>
                    <p>This freelancer has a <strong>{retention_prob:.1f}%</strong> probability of retaining the client.</p>
                    <p><strong>Recommendation:</strong> Consider additional support or alternative freelancers.</p>
                </div>
                """, unsafe_allow_html=True)
            # Probability visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = retention_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Retention Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check your input data and try again.")

def analytics_page():
    st.header("📈 Analytics Dashboard")
    # Sample data for demonstration
    sample_data = {
        'Retention Rate': [85, 78, 92, 88, 76],
        'Rating': [4.5, 4.2, 4.8, 4.6, 4.1],
        'Price Range': ['$0-100', '$100-500', '$500-1000', '$1000-2000', '$2000+'],
        'Projects': [120, 89, 67, 45, 23]
    }
    
    df = pd.DataFrame(sample_data)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(df, x='Price Range', y='Retention Rate', 
                     title='Retention Rate by Price Range',
                     color='Retention Rate',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.scatter(df, x='Rating', y='Retention Rate', 
                         size='Projects', 
                         title='Retention Rate vs Rating',
                         color='Retention Rate',
                         color_continuous_scale='RdYlGn')
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Retention Rate", "83.8%", "2.3%")
    with col2:
        st.metric("Average Rating", "4.44", "0.1")
    with col3:
        st.metric("Total Projects", "344", "12")
    with col4:
        st.metric("High-Risk Projects", "23%", "-1.2%")

def model_info_page():
    st.header("🤖 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Performance")
        performance_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.9978, 0.9955, 1.0000, 0.9975]
        }
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
        # Performance chart
        fig = px.bar(perf_df, x='Metric', y='Value', 
                    title='Model Performance Metrics',
                    color='Value',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(yaxis_range=[0.99, 1.01])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Details")
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        
        **Features Used:**
        - Numerical: Pricing, ratings, review counts
        - Categorical: Geography, currency, rate type
        - Text: Job titles and descriptions (TF-IDF)
        - Tags: Project categories (Multi-label)
        
        **Training:**
        - Dataset: Kaggle Freelancer Dataset
        - Split: 80/20 train-test
        - Validation: 5-fold cross-validation

        **Preprocessing:**
        - Currency normalization to USD
        - Text vectorization with TF-IDF
        - Label encoding for categorical variables
        - One-hot encoding for categorical features
        """)
    st.subheader("🔧 Feature Importance")
    feature_importance = {
        'Feature': ['Average Rating', 'Review Count', 'Price Range', 'Job Description', 
                   'Client Country', 'Tags', 'Currency', 'Rate Type'],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    }
    
    importance_df = pd.DataFrame(feature_importance)
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                orientation='h',
                title='Feature Importance in Retention Prediction',
                color='Importance',
                color_continuous_scale='RdYlGn')
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
