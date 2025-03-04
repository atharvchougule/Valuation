# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import tensorflow as tf
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load dataset
file_path = "abc.xlsx"  # Change this if needed
df = pd.read_excel(file_path, sheet_name="Innovations")

# Select relevant columns
columns_needed = [
    "Innovation Name", "Problem", "Solution", "Innovation Description",
    "Competitive Advantage", "Stage", "Lives impacted",
    "Amount of Funds Raised to Date (in USD)","Success Score"
]
df_selected = df[columns_needed].dropna()

# Encode categorical "Stage"
df_selected["Stage"] = df_selected["Stage"].astype(str)  # Ensure it's a string
label_encoder = LabelEncoder()
df_selected["Stage"] = label_encoder.fit_transform(df_selected["Stage"])
known_labels = set(label_encoder.classes_)  # Store known labels

# Convert "Lives impacted" & "Amount of Funds Raised" to numeric, replacing errors with NaN
df_selected["Lives impacted"] = pd.to_numeric(df_selected["Lives impacted"], errors="coerce").fillna(0)
df_selected["Amount of Funds Raised to Date (in USD)"] = pd.to_numeric(
    df_selected["Amount of Funds Raised to Date (in USD)"], errors="coerce"
).fillna(0)

# Normalize numerical values
scaler = StandardScaler()
df_selected["Lives impacted"] = scaler.fit_transform(df_selected[["Lives impacted"]])
df_selected["Amount of Funds Raised to Date (in USD)"] = scaler.fit_transform(
    df_selected[["Amount of Funds Raised to Date (in USD)"]])

# Convert text features to numerical embeddings using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
text_features = ["Problem", "Solution", "Innovation Description", "Competitive Advantage"]
df_selected["combined_text"] = df_selected[text_features].apply(lambda x: ' '.join(x.astype(str)), axis=1)
X_text = vectorizer.fit_transform(df_selected["combined_text"]).toarray()

# Prepare feature set
X_final = np.hstack((X_text, df_selected[["Stage", "Lives impacted", "Amount of Funds Raised to Date (in USD)"]].values))

# Generate dummy success scores (since dataset lacks one)
y = df_selected["Success Score"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

### Train Multiple Models ###

# 1. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Random Forest MSE:", mse_rf)

# 2. XGBoost Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print("XGBoost MSE:", mse_xgb)



# Compare Models
best_model = min([(mse_rf, "Random Forest"), (mse_xgb, "XGBoost")], key=lambda x: x[0])
print(f"Best Model: {best_model[1]} with MSE: {best_model[0]}")

### Save the best model ###
if best_model[1] == "Random Forest":
    joblib.dump(rf_model, "idea_evaluation_rf.pkl")
elif best_model[1] == "XGBoost":
    joblib.dump(xgb_model, "idea_evaluation_xgb.pkl")


### Train Revenue Prediction Model ###
revenue_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
revenue_model.fit(X_text, df_selected["Amount of Funds Raised to Date (in USD)"])
joblib.dump(revenue_model, "revenue_prediction_xgb.pkl")

### Predict Idea Score ###
def evaluate_new_idea(idea, stage, lives_impacted, funds_raised):
    idea_text_vector = vectorizer.transform([idea]).toarray()
    stage_encoded = label_encoder.transform([stage])[0] if stage in known_labels else -1
    lives_scaled = scaler.transform([[lives_impacted]])[0, 0]
    funds_scaled = scaler.transform([[funds_raised]])[0, 0]
    input_data = np.hstack((idea_text_vector, [[stage_encoded, lives_scaled, funds_scaled]]))

    if best_model[1] == "Random Forest":
        model = joblib.load("idea_evaluation_rf.pkl")
    elif best_model[1] == "XGBoost":
        model = joblib.load("idea_evaluation_xgb.pkl")
    else:
        model = tf.keras.models.load_model("idea_evaluation_nn.h5")

    predicted_score = model.predict(input_data)
    return round(predicted_score[0], 2)

### Predict Required Revenue ###
##def predict_revenue(idea):
    # Convert idea text to numerical features using the same TF-IDF model
    ##idea_text_vector = vectorizer.transform([idea]).toarray()

    # Load the trained revenue prediction model
   ## revenue_model = joblib.load("revenue_prediction_xgb.pkl")  # Now using XGBoost

    # Predict the required funding
    ##predicted_revenue_scaled = revenue_model.predict(idea_text_vector)

    # Convert the prediction back to original scale
    ##predicted_revenue = scaler.inverse_transform([[0, predicted_revenue_scaled[0]]])[0, 1]

    return round(predicted_revenue, 2)

# Example Predictions
idea_description = "A blockchain-based secure voting system"
idea_score = evaluate_new_idea(idea_description, "Scaling", 50000, 200000)

print(f"Predicted Idea Score: {idea_score}/10")

# Save the trained models
joblib.dump(rf_model, "idea_evaluation_rf.pkl")
joblib.dump(xgb_model, "idea_evaluation_xgb.pkl")

# Save the revenue prediction model
joblib.dump(revenue_model, "revenue_prediction_xgb.pkl")

# Save the preprocessing objects
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("All models and preprocessing objects saved successfully!")
