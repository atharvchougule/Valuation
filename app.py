from groq import Groq
import streamlit as st
import joblib
import numpy as np

# Load models and preprocessing objects
rf_model = joblib.load("idea_evaluation_rf.pkl")
revenue_model = joblib.load("revenue_prediction_xgb.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("🚀 AI-Based Idea Evaluation & Revenue Prediction")
st.subheader("Enter your idea details below:")

# Input fields
idea_description = st.text_area("💡 Innovation Description", placeholder="Describe your idea...")
stage = st.selectbox("📌 Stage of Innovation", label_encoder.classes_)
lives_impacted = st.number_input("👥 Estimated Lives Impacted", min_value=0, value=1000)
funds_raised = st.number_input("💰 Funds Raised (in USD)", min_value=0, value=50000)

# Fix input processing
def evaluate_new_idea(idea, stage, lives_impacted, funds_raised):
    idea_vector = vectorizer.transform([idea]).toarray()
    stage_encoded = label_encoder.transform([stage])[0]
    combined_input = np.hstack((idea_vector, [[stage_encoded, lives_impacted, funds_raised]]))
    scaled_input = scaler.transform(combined_input)
    predicted_score = rf_model.predict(scaled_input)
    return round(predicted_score[0], 2)

# Optional: Explain score via Groq API
API_KEY = "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"
client = Groq(api_key=API_KEY)

def generate_success_score(idea_score, idea_description, stage, lives_impacted, funds_raised):
    prompt = (
        f"Predicted Idea Score: {idea_score}/10 for '{idea_description}' at '{stage}' stage. "
        f"Key factors: Lives impacted ({lives_impacted}), Funds raised (${funds_raised}). "
        f"Give **2 short bullet points**:\n"
        f"1️⃣ Why it got this score.\n"
        f"2️⃣ How to improve it.\n"
        f"Keep each point under **30 words**."
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",  # Use a working model
            temperature=0.5,
            max_tokens=100
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"

# Predict button
if st.button("🔍 Predict Idea Success & Revenue"):
    if idea_description.strip():
        idea_score = evaluate_new_idea(idea_description, stage, lives_impacted, funds_raised)
        st.success(f"✅ **Predicted Idea Score:** {idea_score:.2f}/10")
        
        explanation = generate_success_score(idea_score, idea_description, stage, lives_impacted, funds_raised)
        st.markdown(f"📖 **Explanation:** {explanation}")
    else:
        st.warning("⚠️ Please enter a valid idea description.")
