
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv("insurance.csv")

# Encode categorical features
region_le = LabelEncoder()
df['region'] = region_le.fit_transform(df['region'])

sex_ohe = OneHotEncoder(drop='first')
sex_encoded = sex_ohe.fit_transform(df[['sex']]).toarray()
sex_encoded_df = pd.DataFrame(sex_encoded, columns=sex_ohe.get_feature_names_out(['sex']))

smoker_ohe = OneHotEncoder(drop='first')
smoker_encoded = smoker_ohe.fit_transform(df[['smoker']]).toarray()
smoker_encoded_df = pd.DataFrame(smoker_encoded, columns=smoker_ohe.get_feature_names_out(['smoker']))

# Combine encoded features
df_encoded = pd.concat([
    df.drop(['sex', 'smoker'], axis=1).reset_index(drop=True),
    sex_encoded_df.reset_index(drop=True),
    smoker_encoded_df.reset_index(drop=True)
], axis=1)

# Feature-target split
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=123)

# Model training
model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print results
print("Evaluation Metrics:")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")
