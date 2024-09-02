import streamlit as st
import pandas as pd
import numpy as np
import fitz
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model, Model
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the pre-trained components
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

encoder_model = load_model('autoencoder_model.h5')

# Load the latent features
latent_features = np.load('latent_features.npy')

# Load preprocessed job and resume data
jobs_preprocessed_df = pd.read_csv('jobs_ui.csv')
candidates = pd.read_csv('resume_ui.csv')

# Extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text  # Return the raw text without preprocessing

# Preprocess the text (lowercase, remove stopwords, etc.)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Display word cloud based on TF-IDF scores
def display_wordcloud(tfidf_matrix, feature_names, num_words=20):
    # Extract top words based on TF-IDF scores
    top_n_ids = np.argsort(tfidf_matrix.toarray()).flatten()[-num_words:]
    top_words = {feature_names[i]: tfidf_matrix[0, i] for i in top_n_ids}
    
    # Generate and display the word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color='white'
        ).generate_from_frequencies(top_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    # Also display the words in the sidebar
    st.sidebar.subheader("Top 20 Relevant Words:")
    for word in sorted(top_words, key=top_words.get, reverse=True):
        st.sidebar.write(f"{word} ({top_words[word]:.4f})")

# Recommend jobs for a given resume
def recommend_jobs_for_resume(new_resume, top_n=10):
    new_resume_tfidf = tfidf_vectorizer.transform([new_resume]).toarray()
    new_resume_encoded = encoder_model.predict(new_resume_tfidf)
    similarities = cosine_similarity(new_resume_encoded, latent_features[:jobs_preprocessed_df.shape[0]])
    top_n_indices = np.argsort(similarities.flatten())[::-1][:top_n]
    return jobs_preprocessed_df.iloc[top_n_indices]

# Recommend resumes for a given job description
def recommend_resumes_for_job(new_job_description, top_n=10):
    new_job_tfidf = tfidf_vectorizer.transform([new_job_description]).toarray()
    new_job_encoded = encoder_model.predict(new_job_tfidf)
    similarities = cosine_similarity(new_job_encoded, latent_features)
    top_n_indices = np.argsort(similarities.flatten())[::-1][:top_n]
    return candidates.iloc[top_n_indices]

# Streamlit app layout
st.title("Job and Resume Recommendation System")

tab1, tab2 = st.tabs(["Resume Upload", "Job Description Input"])

with tab1:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Upload a Resume (PDF)", type="pdf")
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        
        # Preprocess the text to remove stopwords
        resume_text_cleaned = preprocess_text(resume_text)
        
        # Display word cloud and words in sidebar for top 20 words
        resume_tfidf = tfidf_vectorizer.transform([resume_text_cleaned])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        st.sidebar.header("Top 20 Words")
        display_wordcloud(resume_tfidf, feature_names, num_words=20)
        
        # Preprocess text and recommend jobs
        resume_text_preprocessed = preprocess_text(resume_text_cleaned)
        top_jobs = recommend_jobs_for_resume(resume_text_preprocessed, top_n=10)
        st.write("Top 10 Job Recommendations:")
        st.dataframe(top_jobs)

with tab2:
    st.header("Enter Job Description")
    job_description = st.text_area("Paste a Job Description Here")
    if job_description:
        # Preprocess the text to remove stopwords
        job_description_cleaned = preprocess_text(job_description)
        
        # Display word cloud and words in sidebar for top 20 words
        job_tfidf = tfidf_vectorizer.transform([job_description_cleaned])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        st.sidebar.header("Top 20 Words")
        display_wordcloud(job_tfidf, feature_names, num_words=20)
        
        # Preprocess text and recommend resumes
        job_description_preprocessed = preprocess_text(job_description_cleaned)
        top_resumes = recommend_resumes_for_job(job_description_preprocessed, top_n=10)
        st.write("Top 10 Resume Recommendations:")
        st.dataframe(top_resumes)
        