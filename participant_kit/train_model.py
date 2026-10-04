"""
Neural Nexus -- Round 2: The Trading Pit
train_model.py

This script demonstrates how to train a basic Machine Learning model offline
using the provided history CSV, and export it as a .pkl file for your agent to use.

CHALLENGE: This is a very basic Random Forest using only raw prices. 
To win, you should engineer better features (rolling averages, volatility, 
cross-asset correlations) and tune a stronger model (XGBoost, MLP, etc.).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load the dataset (ensure this matches the CSV provided in your kit)
DATA_FILE = "history.csv"
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Could not find {DATA_FILE}. Make sure it is in the same folder.")
    exit()

ASSETS = ["quantum_dynamics", "byte_stream", "gold_trust", 
          "metro_rail", "agro_futures", "solar_grid"]

print("Training models for each asset. This may take a moment...")

models = {}

for asset in ASSETS:
    # Get just the prices for this specific asset
    prices = df[df['asset'] == asset]['price'].dropna().values
    X, y = [], []
    
    # TODO:
    # =======================================================================
    # FREEDOM TO EDIT: FEATURE ENGINEERING & DATA SHAPING
    # =======================================================================
    # - You DO NOT have to stick to a 20-round sliding window. You can 
    #   change it to 5, 50, or whatever window size you think works best.
    # - You can engineer new features. Instead of just appending raw prices, 
    #   calculate price differences, rolling averages, volatility, or momentum.
    # - You can include prices from OTHER assets (cross-asset correlations) 
    #   if you think QNTM's price movement affects BYTE's price.
    # =======================================================================
    # Dummy Split:
    WINDOW_SIZE = 20
    for i in range(WINDOW_SIZE, len(prices) - 1):
        X.append(prices[i-WINDOW_SIZE : i])
        y.append(prices[i])
        
    X = np.array(X)
    y = np.array(y)
    
    # TODO:
    # =======================================================================
    # FREEDOM TO EDIT: MODEL SELECTION & HYPERPARAMETERS
    # =======================================================================
    # - You can swap out RandomForestRegressor for XGBoost, LinearRegression,
    #   a PyTorch Neural Network, SVM, or any other model.
    # - You can tune hyperparameters (e.g., n_estimators=100, deeper trees).
    # - You can even train DIFFERENT types of models for DIFFERENT assets!
    #   (e.g., models["quantum_dynamics"] = XGBoost(...), models["gold_trust"] = LinearRegression(...))
    # =======================================================================
    # Dummy Model (Training an rf model for each asset and adding to models dict)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X, y)
    models[asset] = rf
    print(f"  - Trained model for {asset}")

# Do Not Change this unless you know what you are doing:
# 2. Export the trained models to a file
# We save a dictionary of 6 separate models (one for each asset).
joblib.dump(models, "model.pkl")

print("\nSuccess! Saved to model.pkl. Your agent.py can now load this file.")
