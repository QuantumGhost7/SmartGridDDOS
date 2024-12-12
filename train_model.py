import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model():
    # Load only the training dataset
    df = pd.read_csv('train_data.csv')
    
    # Prepare features and target
    X = df[['packets_per_second', 'speed_mbps', 'packet_size']]
    y = df['is_ddos']
    
    # Split the training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Validate the model
    print("Validation Results:")
    y_val_pred = model.predict(X_val_scaled)
    print(classification_report(y_val, y_val_pred))
    
    # Save the model and scaler
    joblib.dump(model, 'ddos_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
if __name__ == "__main__":
    train_model()