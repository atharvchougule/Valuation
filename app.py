from groq import Groq
import re
import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# Load models and preprocessing objects
rf_model = joblib.load("idea_evaluation_rf.pkl")
xgb_model = joblib.load("idea_evaluation_xgb.pkl")
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

# Prediction function
def evaluate_new_idea(idea, stage, lives_impacted, funds_raised):
    idea_text_vector = vectorizer.transform([idea]).toarray()
    stage_encoded = label_encoder.transform([stage])[0]
    lives_scaled = scaler.transform([[lives_impacted]])[0, 0]
    funds_scaled = scaler.transform([[funds_raised]])[0, 0]
    input_data = np.hstack((idea_text_vector, [[stage_encoded, lives_scaled, funds_scaled]]))

    best_model = rf_model  # Change to rf_model or nn_model if required
    predicted_score = best_model.predict(input_data)

    return round(predicted_score[0], 2)

# Load Groq API Key
API_KEY = "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"
client = Groq(api_key=API_KEY)

# Function to explain the score
def generate_success_score(idea_score, idea_description, stage, lives_impacted, funds_raised):
    prompt = (
        f"My Predicted Idea Score is {idea_score} for the idea: '{idea_description}' "
        f"in the stage '{stage}', which has impacted {lives_impacted} lives and raised ${funds_raised}.\n"
        f"Explain why my model might have given {idea_score} based on these factors."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=100
        )

        explanation = chat_completion.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"Error generating explanation: {e}"


# Predict button
if st.button("🔍 Predict Idea Success & Revenue"):
    if idea_description.strip():
        idea_score = evaluate_new_idea(idea_description, stage, lives_impacted, funds_raised)
        st.success(f"✅ **Predicted Idea Score:** {idea_score:.2f}/10")
        
        # Generate explanation and display it
        explanation = generate_success_score(idea_score, idea_description, stage, lives_impacted, funds_raised)
        st.write(f"📖 **Explanation:** {explanation}")

    else:
        st.warning("⚠️ Please enter a valid idea description.")





