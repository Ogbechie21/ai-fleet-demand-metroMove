AI-Based Fleet Performance & Risk Prediction System
Project Overview

This project develops a machine learning system to analyze fleet operations for MetroMove Transit Services. The system performs:

Classification: Predicts whether a trip is high-risk (delayed)

Regression: Predicts maintenance cost

Cloud Deployment: Deployed using Streamlit

Dataset

Source: Kaggle – Fleet Dataset
File: fleet_dummy_5000.csv

Features include:

Trip details

GPS locations

Vehicle information

Behavioral indicators

Cost metrics

Weather conditions

Machine Learning Tasks
Classification

Binary classification to label trips as:

Normal

High risk (delay)

Regression

Predict operational maintenance cost.

Project Structure
ai-fleet-demand-metroMove
│
├── data/                # Dataset
├── notebooks/           # Analysis notebooks
├── report/              # Final report
├── streamlit_app.py     # Deployed app
├── requirements.txt     # App dependencies
└── README.md

Running Locally
Step 1 — Install dependencies
pip install -r requirements.txt

Step 2 — Run Streamlit app
streamlit run streamlit_app.py

Live Application

Public web app:

https://ai-fleet-demand-metromove-o8qov4sdpgtreftqbj6ysu.streamlit.app/

Author

Christopher F. Ogbechie
ANLT202