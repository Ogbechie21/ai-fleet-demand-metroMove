## AI-Based Fleet Performance & Risk Prediction System

This project develops a machine learning system designed to improve fleet operations for MetroMove Transit Services. The goal is to help the organization make better decisions regarding scheduling, routing, and maintenance planning. The system focuses on three core capabilities. First, it includes a classification model that predicts whether a trip is likely to become high-risk or delayed. Second, it includes a regression model that estimates the expected maintenance cost for a trip or vehicle. Third, the models are deployed through a Streamlit web application, allowing users to access predictions online in a simple and interactive format.

## Dataset
The dataset used in this project was obtained from Kaggle and is stored in the file named “fleet_dummy_5000.csv.” It contains a wide variety of operational information related to MetroMove’s fleet activities. The data includes trip details such as distance and duration, GPS coordinates from the start and end of trips, vehicle-related information, driver behavior indicators such as violations and speeding events, financial information such as fuel cost, toll cost, and driver pay, as well as time-related fields like pickup hour and day of the week. Weather conditions are also included. Together, these features make the dataset suitable for both classification and regression tasks.

## Machine Learning Tasks
Classification
The classification component labels each trip as either “Normal” or “High Risk.” A trip is considered high risk when it is predicted to be delayed. This helps MetroMove respond proactively to operational challenges, improve scheduling efficiency, and reduce unexpected delays.

Regression
The regression component predicts the maintenance cost associated with a particular trip or vehicle. This supports maintenance planning, budget forecasting, and identifying operational patterns that may lead to increased repair expenses.

## Project Structure
The project is organized into a folder named “ai-fleet-demand-metroMove.” Inside this folder, several key components are included. The “data” folder contains the dataset used throughout the project. The “notebooks” folder includes notebook files used for exploratory data analysis, preprocessing, and model experimentation. The “report” folder holds the final written report summarizing the findings and results. The “streamlit_app.py” file runs the Streamlit application, which provides the user interface for predictions. A “requirements.txt” file lists all dependencies needed to run the project. The “README.md” file provides documentation and instructions for users.

## Running the Project Locally
To run the system locally, you first need to install all required dependencies. This is done by opening a terminal and running the command: pip install -r requirements.txt. After the installation is complete, the Streamlit application can be launched by running the command: streamlit run streamlit_app.py. Once launched, the application will open in a browser window where users can enter trip and vehicle information to receive predictions.

## Live Application
A public version of the web application is available online. It can be accessed directly through the following link:

https://ai-fleet-demand-metromove-o8qov4sdpgtreftqbj6ysu.streamlit.app/

This live version allows users to test the prediction system without installing anything.

## Authors
Lukman Ibrahim
Christopher F. Ogbechie
Asare Matthew
ANLT202

