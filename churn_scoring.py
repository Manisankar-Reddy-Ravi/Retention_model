import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 📥 Step 1: Load data
df = pd.read_csv("Churn_time.csv")

# 🎯 Step 2: Define features and target
target = "Churned"
features = [
    'Customer_ID', 'Address_Change_Flag',
    'VIN_Validated', 'Policy_Tenure_Months', 'Coverage_Type',
    'Deductibles', 'Has_Multi_Policy', 'Loyalty_Program_Enrollment',
    'Billing_Method', 'Payment_Method', 'Discount_Count',
    'Premium_Change_Percent_Last_Renewal', 'Late_Payment_Count', 'Auto_Renew_Enabled',
    'Claims_Count_Lifetime', 'At_Fault_Accident_Count', 'Claim_Satisfaction_Score',
    'Interaction_Score', 'NPS', 'Complaint_Count', 'Sentiment_Score'
]

# 🧹 Step 3: Drop ID and handle missing
df = df[features + [target]].copy()
df.drop(columns=["Customer_ID"], inplace=True)
df.fillna(0, inplace=True)

# 🔣 Step 4: Encode categoricals and save encoders
encoders = {}
for col in ['Coverage_Type', 'Billing_Method', 'Payment_Method']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
os.makedirs("models", exist_ok=True)
joblib.dump(encoders, "models/label_encoders.pkl")
print("✅ Encoders saved to models/label_encoders.pkl")

# 📊 Step 5: Train-test split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Step 6: Train model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# 📈 Step 7: Evaluate & save
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))

# 💾 Save model
joblib.dump(model, "models/xgb_model.pkl")
print("✅ Model saved to models/xgb_model.pkl")
