import streamlit as st
import joblib

# Load the TF-IDF vectorizer and the model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Function to classify the input text
def classify_text(text):
    tfidf_text = vectorizer.transform([text])
    prediction = model.predict(tfidf_text)
    return prediction[0]

# Streamlit app
def main():
    st.title("Spam Email Classification App")

    # Input text box for user to enter email content
    user_input = st.text_area("Enter the email content:", "")

    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Perform classification
            prediction = classify_text(user_input)

            # Display the prediction
            if prediction == "spam":
                st.error("This email is classified as SPAM.")
            else:
                st.success("This email is classified as NON-SPAM.")

if __name__ == "__main__":
    main()
