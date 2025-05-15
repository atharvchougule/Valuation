from groq import Groq
import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# Load models and preprocessing objects

rf\_model = joblib.load("idea\_evaluation\_rf.pkl")
xgb\_model = joblib.load("idea\_evaluation\_xgb.pkl")
revenue\_model = joblib.load("revenue\_prediction\_xgb.pkl")

vectorizer = joblib.load("tfidf\_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
label\_encoder = joblib.load("label\_encoder.pkl")

# Streamlit UI

st.title("🚀 AI-Based Idea Evaluation & Revenue Prediction")
st.subheader("Enter your idea details below:")

# Input fields

idea\_description = st.text\_area("💡 Innovation Description", placeholder="Describe your idea...")
stage = st.selectbox("📌 Stage of Innovation", label\_encoder.classes\_)
lives\_impacted = st.number\_input("👥 Estimated Lives Impacted", min\_value=0, value=1000)
funds\_raised = st.number\_input("💰 Funds Raised (in USD)", min\_value=0, value=50000)

# Prediction function

def evaluate\_new\_idea(idea, stage, lives\_impacted, funds\_raised):
idea\_text\_vector = vectorizer.transform(\[idea]).toarray()
stage\_encoded = label\_encoder.transform(\[stage])\[0]
lives\_scaled = scaler.transform(\[\[lives\_impacted]])\[0, 0]
funds\_scaled = scaler.transform(\[\[funds\_raised]])\[0, 0]
input\_data = np.hstack((idea\_text\_vector, \[\[stage\_encoded, lives\_scaled, funds\_scaled]]))

```
best_model = rf_model  # Change to rf_model or nn_model if required
predicted_score = best_model.predict(input_data)

return round(predicted_score[0], 2)
```

# Load Groq API Key

API\_KEY = "gsk\_PCUO1PQlPITuGlsWhpZQWGdyb3FYd2SQtg1NSlS1ooZBxke63Niy"
client = Groq(api\_key=API\_KEY)

# Function to explain the score

def generate\_success\_score(idea\_score, idea\_description, stage, lives\_impacted, funds\_raised):
prompt = (
f"Predicted Idea Score: {idea\_score}/10 for '{idea\_description}' at '{stage}' stage. "
f"Key factors: Lives impacted ({lives\_impacted}), Funds raised (\${funds\_raised}). "
f"Give **2 short bullet points**: "
f"1️⃣ Why it got this score. "
f"2️⃣ How to improve it. "
f"Keep each point under **30 words**."
)

```
try:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mistral-saba-24b",
        temperature=0.5,
        max_tokens=100
    )

    explanation = chat_completion.choices[0].message.content.strip()
    return explanation
except Exception as e:
    return f"Error generating explanation: {e}"
```

# Predict button

if st.button("🔍 Predict Idea Success & Revenue"):
if idea\_description.strip():
idea\_score = evaluate\_new\_idea(idea\_description, stage, lives\_impacted, funds\_raised)
st.success(f"✅ **Predicted Idea Score:** {idea\_score:.2f}/10")

```
    # Generate explanation and display it
    explanation = generate_success_score(idea_score, idea_description, stage, lives_impacted, funds_raised)
    st.write(f"📖 **Explanation:** {explanation}")

else:
    st.warning("⚠️ Please enter a valid idea description.")
```
