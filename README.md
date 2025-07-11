# 📊 Client Retention Prediction Based on Client Behavior

This project uses machine learning to predict whether clients on freelancing platforms are likely to return, based on their behavioral patterns, job posting characteristics, and engagement history.

## 🚀 Overview

The model is trained on a real-world freelancer job dataset and leverages features like:
- Job title and description (TF-IDF)
- Tags (MultiLabelBinarizer)
- Ratings and review counts
- Pricing (converted to USD)
- Geographic data (encoded)

It achieves a **high F1-score of 0.9975** using a `RandomForestClassifier`, making it ideal for real-world deployment in client retention and engagement systems.

---

## 🧠 Model Details

- **Algorithm**: RandomForestClassifier (scikit-learn)
- **Validation**: 5-Fold Cross-Validation
- **Target**: Binary classification (Client Retained or Not)
- **Performance**:
  - Accuracy: 99.78%
  - F1-Score: 0.9975
  - Precision: 99.55%
  - Recall: 100%

---
## 🔍 NLP Features

This project leverages Natural Language Processing (NLP) techniques to extract and utilize insights from the textual data in job listings:

- **TF-IDF Vectorization**  
  - Applied to job titles and job descriptions  
  - Captures the importance of words relative to all listings  
  - Helps the model differentiate between common and domain-specific terms

- **Text Preprocessing**  
  - Lowercasing
  - Removal of punctuation and stopwords
  - Tokenization using NLTK  
  - Ensures consistent and clean input for vectorization

- **MultiLabel Tag Encoding**  
  - Job tags are split, cleaned, and encoded using `MultiLabelBinarizer`  
  - Converts variable-length tag lists into fixed-size feature vectors

These NLP features allow the model to understand the nature and content of job postings, which significantly improves prediction accuracy regarding client retention behavior.
---

## 📈 Visual Insights

- **Top 10 Client Countries**
- **Top 20 Most Frequent Tags**
- Interactive dashboards and visualizations powered by Streamlit.

---

## 🛠️ Tech Stack

- Python
- scikit-learn
- pandas / numpy
- NLTK
- Plotly / Seaborn
- Streamlit
- Fixer.io API (currency conversion)

---

## 📊 Streamlit Dashboard

Explore the project live and make predictions interactively:

👉 [Streamlit App](https://client-retention-prediction.streamlit.app/)

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/RehamHassan1/Client-Retention-Prediction.git
cd Client-Retention-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
