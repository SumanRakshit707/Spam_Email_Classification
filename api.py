from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer and the model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Function to classify the input text
def classify_text(text):
    tfidf_text = vectorizer.transform([text])
    prediction = model.predict(tfidf_text)
    return prediction[0]

# FastAPI app
app = FastAPI(title='Spam Email Classification')

# Request model
class EmailContent(BaseModel):
    content: str

# Route to handle classification
@app.post("/classify/")
def classify_email(email: EmailContent):
    text = email.content

    if not text.strip():
        return {"error": "Please provide some text."}

    # Perform classification
    prediction = classify_text(text)

    # Return the prediction
    if prediction == "spam":
        return {"classification": "SPAM"}
    else:
        return {"classification": "NON-SPAM"}
# uvicorn app:app --reload
