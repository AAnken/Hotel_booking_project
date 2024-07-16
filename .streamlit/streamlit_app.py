import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime, date, time, timedelta 
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder
import altair as alt


# load data
BASE_DIR = Path(__file__).resolve(strict=True).parents[1]
#file_path = '../data/hotel_bookings.csv'
with open(f"{BASE_DIR}/data/hotel_bookings.csv", 'rb') as f:
    data = pd.read_csv(f)

countries = list(data.country.unique())
agent = list(data.agent.sort_values().unique())
travel_type = list(data.customer_type.unique())
room_type = list(data.reserved_room_type.sort_values().unique())
market_segment = list(data.market_segment.unique())

col1, col2 = st.columns(2)
col1.image(f"{BASE_DIR}/images/City_Hotel.jpeg", width = 400)
col2.image(f"{BASE_DIR}/images/Resort_Hotel.jpeg", width = 400)
st.title('Hotel Ancora Branca - City and Resort Hotel')


st.subheader('Book your room')

country = st.selectbox('Enter country of origin', countries)

hotel = st.selectbox('choose your hotel', ['City Hotel','Resort Hotel'])

deposit = 'No Deposit'

col1, col2 = st.columns(2)
arrival_date = col1.date_input('Arrival')

departure_date = col2.date_input('Departure')

duration_nights = departure_date.toordinal() - arrival_date.toordinal()

arrival_week = arrival_date.isocalendar().week

arrival_day = arrival_date.timetuple().tm_yday

lead_time = arrival_date.toordinal() - date.today().toordinal()

# count weekend nights
weekend_nights = 0
week_nights = 0
current_date = arrival_date
for i in range(0,duration_nights):
    if current_date.weekday() == 5:
        weekend_nights += 1
    elif current_date.weekday() == 6:
        weekend_nights += 1
    else:
        week_nights += 1
    current_date += timedelta(days=1)

# Guests
col1, col2, col3 = st.columns(3)

adults = col1.number_input('No. of adults', 0, 15)
children = col2.number_input('No. of children', 0, 15)
babies = col3.number_input('No. of babies', 0, 15)
children_total = children + babies

if duration_nights < 2:
    st.write(f' Your booking request is for 1 night')
else:
    st.write(f' Your booking request is for {duration_nights} nights')

meal = st.selectbox('choose your meal plan', ['BB', 'FB', 'HB', 'SC'])

room = st.selectbox('Choose your room type', room_type)

car = st.number_input('Required car parking lot',0,10)

agent = st.selectbox('If applicable, select travel agency code', agent)

customer_type = st.selectbox('Please select type of travel', travel_type)

market = st.selectbox('Please select type of market', market_segment)

request = st.multiselect('please indicate special requests:', ['Baby bed', 'Dog', 'Cat', 'Extra blanket', 'Quiet room'])

total_requests = len(request)


###################################### Predict adr based on adr model and inputs
adr_columns = ['hotel', 'meal', 'market_segment', 'reserved_room_type', 'lead_time',
       'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
       'babies', 'agent', 'required_car_parking_spaces',
       'total_of_special_requests', 'arrival_day']


# load LabelEncoder
with open('.streamlit/model_adr/label_hotel.pkl', "rb") as f:
    le_hotel = pickle.load(f)

with open('.streamlit/model_adr/label_meal.pkl', "rb") as f:
    le_meal = pickle.load(f)

with open('.streamlit/model_adr/label_market.pkl', "rb") as f:
    le_market = pickle.load(f)

with open('.streamlit/model_adr/label_room.pkl', "rb") as f:
    le_room = pickle.load(f)

#load scaler
with open('.streamlit/model_adr/scaler_adr.pkl', "rb") as f:
    scaler_adr = pickle.load(f)

#load model
with open('.streamlit/model_adr/ML_adr.pkl', "rb") as f:
    model_adr = pickle.load(f)

# transform categorical inputs
hotel_a = le_hotel.transform([hotel])
meal_a = le_meal.transform([meal])
market_a = le_market.transform([market])
room_a = le_room.transform([room])

# create Dataframe
adr_df = pd.DataFrame(columns=adr_columns,data=[[hotel_a,meal_a,market_a,room_a,lead_time,weekend_nights,week_nights,adults,children,babies,agent,car,total_requests,arrival_day]])

# scale dataframe
X_adr = scaler_adr.transform(adr_df)

# predict on adr model
adr_output = model_adr.predict(X_adr)

###################################### Predict cancelation
cancel_columns = ['hotel', 'meal', 'country', 'market_segment', 'reserved_room_type',
       'deposit_type', 'lead_time', 'arrival_date_week_number', 'adults',
       'children', 'babies', 'agent', 'adr', 'required_car_parking_spaces',
       'total_of_special_requests', 'children_all', 'duration_of_stay']

# load label encoder
with open('.streamlit/model_cancel/label_hotel.pkl', "rb") as f:
    len_hotel = pickle.load(f)

with open('.streamlit/model_cancel/label_meal.pkl', "rb") as f:
    len_meal = pickle.load(f)

with open('.streamlit/model_cancel/label_country.pkl', "rb") as f:
    len_country = pickle.load(f)

with open('.streamlit/model_cancel/label_market.pkl', "rb") as f:
    len_market = pickle.load(f)

with open('.streamlit/model_cancel/label_room.pkl', "rb") as f:
    len_room = pickle.load(f)

with open('.streamlit/model_cancel/label_deposit.pkl', "rb") as f:
    len_deposit = pickle.load(f)

#load scaler
with open('.streamlit/model_cancel/scaler_cancel.pkl', "rb") as f:
    scaler_cancel = pickle.load(f)

#load model
with open('.streamlit/model_cancel/ML_cancel.pkl', "rb") as f:
    model_cancel = pickle.load(f)

# transform categorical inputs
hotel_c = len_hotel.transform([hotel])
meal_c = len_meal.transform([meal])
country_c = len_country.transform([country])
market_c = len_market.transform([market])
room_c = len_room.transform([room])
deposit_c = len_deposit.transform([deposit])

# create Dataframe
cancel_df = pd.DataFrame(columns=cancel_columns,
                         data=[[hotel_c,meal_c,country_c, market_c,room_c,deposit_c,
                                lead_time, arrival_week,adults,children,babies,agent,
                                adr_output,car,total_requests,children_total,duration_nights]])

# scale dataframe
X_cancel = scaler_cancel.transform(cancel_df)

# predict on adr model
cancel_output = model_cancel.predict(X_cancel)

############################################## Show adr and cancellation condition
#Information for Hotel manager
st.subheader('*Information for Hotel Manager*')
if cancel_output == 0:
    st.write('Machine learning algorithm predicts no cancelation')
else:
    st.write('Machine learning algorithm predicts cancelation!  \nDeposit required!  \nCancelation period is 7 days')

# Information for guest
col1, col2 = st.columns(2)
col1.subheader('Book your room at the best condition')
col1.write(f'Your rate per day is: {float(adr_output):.2f}€')

if duration_nights <4:
    col1.write('**We take sustainability serious!**  \nFor stays no longer than 3 nights, room service is not included. This helps to save water and energy resources')
    room_service = col1.radio('Do you want to add daily room service?', ['no', 'yes'])
else:
    col1.write('**We take sustainability serious!**  \nTo reduce carbon emission, we offer room service every third day as a standard.')
    room_service = col1.radio('Do you want to add daily room service?', ['no', 'yes'])

if room_service == 'no':
    col1.write(f'**Book your room now for {float(adr_output*duration_nights):.2f}€**')
else:
    col1.write(f'**Book your room now for {float(adr_output*duration_nights*1.1):.2f}€**')

if cancel_output == 0:
    col1.write('*Your cancelation period is 1 day*')
elif cancel_output == 1 and lead_time < 7:
    col1.write('*No cancelation possible.  \nA deposit is required to confirm your reservation*')
elif cancel_output == 1 and lead_time >= 7:
    col1.write('*Cancelation period is 7 days.  \nA deposit is required to confirm your reservation*')

book = col1.button('BOOK NOW')

# carbon emission in kg CO2e per room per night
carbon_emission = 19

carbon_total = carbon_emission*duration_nights

if room_service == 'no':
    carbon_total = carbon_total*0.8
    carbon = {'carbon total': carbon_total}
elif room_service == 'yes':
    carbon_total = carbon_total
    carbon = {'carbon total': carbon_total}

compensation_rate = 5/70

carbon_df = pd.DataFrame(columns=['carbon total'], data=[[carbon_total]])

col2.subheader('Make your stay more sustainable!')
col2.write('check your carbon footprint for your stay at our hotel:')
col2.bar_chart(data=carbon_df,y='carbon total',y_label='kg CO2 equivalent', x_label='', color='#490648')
col2.write(f"Compensate your carbon footprint for your stay by donating {float(carbon_total*compensation_rate):.2f}€ with our partner ForTomorrow.eu")
col2.link_button('Donate', "https://www.fortomorrow.eu/en/donate-for-climate-protection")
