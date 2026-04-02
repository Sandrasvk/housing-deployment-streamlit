import streamlit as st 
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title = 'Estate Predictor', page_icon = '🏠', layout = 'wide')

model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.header('House Features')
st.info('Adjust the sliders to match the area statistics')

with st.sidebar:
    income = st.slider('Avg. Area Income', 17000.0, 110000.0, 68000.0, step = 500.0)
    avg_area_house = st.slider('Avg. Area House Age', 2.0,10.0,6.0,step = 0.1)
    room_area = st.slider('Avg. Area Number of rooms', 3.0, 11.0, 7.0, step = 0.1)
    bedroom_area = st.slider('Avg. Area Number of Bedrooms', 2.0, 7.0, 4.0, step = 0.1)
    area_pop = st.number_input('Area Population', value = 36000.0)


st.title('🏠House Price Prediction')
st.markdown('---------')

col1,col2 = st.columns([1.5,1])
with col1:
    st.image('housing prediction img.WEBP', use_column_width = True)

with col2:
    st.subheader('Price Estimation')
    st.write('Click the button below to generate a prediction based on sidebar values')

if st.button('Predict Price'):
    input_feature = np.array([[income, avg_area_house, room_area, bedroom_area, area_pop]])
    scaled_feature = scaler.transform(input_feature)
    prediction = model.predict(scaled_feature)
    st.balloons()
    st.metric(label = "Predicted Price", value = f"${prediction[0]:,.2f}")
    st.success('Prediction generated successfully!')