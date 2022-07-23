import os
import pickle
import streamlit as st
import sys
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer, WordNetLemmatizer



model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
model_path = os.path.join(sys.path[0], model_name)
vect_path = os.path.join(sys.path[0], vectorizer_name)
model = pickle.load(open(model_name, 'rb'))
vect = pickle.load(open(vectorizer_name, 'rb'))

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer= WordNetLemmatizer()

def clean(review):
    review = ' '.join([word for word in review.split() if word not in (stop_words)])
    review = ' '.join([stemmer.stem(word) for word in review.split()])
    review = ' '.join([lemmatizer.lemmatize(word) for word in review.split()])
    return review

def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = clean(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"


def main():
    st.markdown("<h1 style='text-align: center; color: White;background-color:#0E1117'>Classifier of Food Reviews</h1>", unsafe_allow_html=True)
    review = st.text_input(label='Write Your Review')
    if st.button('Classify'):
        result = raw_test(review, model, vect)
        st.success(
            'This Review Is {}'.format(result))


if __name__ == '__main__':
    main()
