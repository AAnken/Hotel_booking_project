# Final Project on Hotel Bookings
Repository for final project within the Data Analytics/ Data Science course at Greenbootcamps.com

Contributors:
- JacquelineWeiss
- AAnken

## Description of Dataset

The dataset can be found on kaggle via this link https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data

This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.

An explanation to the features can be found [here](https://github.com/AAnken/Hotel_booking_project/blob/main/column_names.md)

## Motivation
The goal is to use the dataset to predict the event of cancellation and forecast the average daily rate of hotel rooms based on guest data. Different machine learning algorithms were compared and the best performing ones were implemented in a streamlit app. A dashboard was created to visualize the main findings of this dataset. 

## Project steps
1. Exploratory data analysis and data cleaning: [EDA](https://github.com/AAnken/Hotel_booking_project/blob/main/EDA.ipynb)
2. Create Tableau Dashboard: [Dashboard](https://public.tableau.com/app/profile/jacqueline.wei./viz/Hotel_Booking_Demand_17207037521430/HotelBookingStory?publish=yes)
3. Compare machine learning models for prediction of cancellation: [ML classification](https://github.com/AAnken/Hotel_booking_project/blob/main/ML_classification.ipynb)
4. Train a Random Forest classifier with a reduced dataset: [RFC](https://github.com/AAnken/Hotel_booking_project/blob/main/ML_reduced_RandomForest.ipynb)
5. Compare machine learning models on reduced dataset to predict adr: [Linear Regression](https://github.com/AAnken/Hotel_booking_project/blob/main/ML_LinearRegression.ipynb)
6. Create a streamlit app with implemented ML models for prediction of cancellation and adr: [Streamlit App](https://github.com/AAnken/Hotel_booking_project/blob/main/.streamlit/streamlit_app.py) (not deployed)

## Set-up virtual environment
```pyenv local 3.11.2```<br>
```python -m venv .venv```<br>
```source .venv/bin/activate```<br>
```pip install --upgrade pip```<br>
```pip install -r requirements.txt```

### Create own git branch
```git switch -c <Name>```