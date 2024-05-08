import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model from the pickle file
with open('Spam_Mail.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('TFIDF_Vectorizer_Spam_Mail.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Main Streamlit app
def main():
    # Title of the app
    st.markdown('<h1 style="color: lightblue; text-align: center; margin-bottom:100px;">Email Spam Detection</h1>', unsafe_allow_html=True)

    # Input form for user input
    input_mail = st.text_area("Type your email message here and press Enter")

    if st.button("Check Spam"):
        # Convert text to feature vector using TF-IDF vectorizer
        input_data_features = tfidf_vectorizer.transform([input_mail])

        # Make prediction
        prediction = loaded_model.predict(input_data_features)

        # Display prediction result with colored text
        if prediction[0] == 1:
            result_text = '<span style="color: green; font-size: 20px;">This email is classified as HAM</span>'
        else:
            result_text = '<span style="color: red; font-size: 20px;">This email is classified as SPAM</span>'
        
        st.markdown(f'<p style="text-align: center; color: gray;">Prediction: {result_text}</p>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()
