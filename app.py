import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="📨", layout="centered")

st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#6A5ACD;">📨 Email/SMS Spam Classifier 🌷</h1>
        <p style="font-size:17px;">Built using Streamlit & Machine Learning 🌸</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input box
input_sms = st.text_area("Enter the message below 👇", height=150)

# Predict button
if st.button('🔍 Detect Spam'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        # Display result with confidence
        if result == 1:
            st.error("🚫🥀 **This message is SPAM!**")
            st.metric(label="Confidence", value=f"{proba[1]*100:.2f}%")
        else:
            st.success("✅🌹**This message is NOT spam.**")
            st.metric(label="Confidence", value=f"{proba[0]*100:.2f}%")

        # Expandable section for processed text
        with st.expander("🔍 View Preprocessed Text"):
            st.code(transformed_sms)


