import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Title
st.title("Predict Total Alcohol Consumption")
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://thumbs.dreamstime.com/b/bottles-famous-global-beer-brands-poznan-pol-mar-including-heineken-becks-bud-miller-corona-stella-artois-san-miguel-143170440.jpg'
             width='400'>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model
with open("beer_serv_model.pkl", "rb") as f:
    beer_serv_model = pickle.load(f)

# Load encoders
with open("country_encoder.pkl", "rb") as file:
    country_encoder = pickle.load(file)

with open("continent_encoder.pkl", "rb") as file:
    continent_encoder = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Inputs
beer_serv = st.slider("Beer Servings", 0, 500, 50)

spirit_serv = st.slider("Spirit Servings", 0, 500, 50)

wine_serv = st.slider("Wine Servings", 0, 500, 50)



continent = st.selectbox(
    "Select Continent",
    ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"]
)

country = st.text_input("Enter Country")

# Predict
if st.button("Predict"):

    try:

        # Encode categorical columns
        country_encoded = country_encoder.transform([country])[0]

        continent_encoded = continent_encoder.transform([continent])[0]

        # Scale numerical data
        numerical_data = np.array([
            [beer_serv, spirit_serv, wine_serv]
        ])

        scaled_data = scaler.transform(numerical_data)

        # Final input
        final_input = np.array([[
            scaled_data[0][0],
            scaled_data[0][1],
            scaled_data[0][2],
            country_encoded,
            continent_encoded
        ]])

        # Prediction
        prediction = beer_serv_model.predict(final_input)

        st.success(
            f"Predicted Total Litres of Pure Alcohol: {prediction[0]:.2f}"
        )

    except Exception as e:
        st.error(f"Error: {e}")

# User values
beer = beer_serv
spirit = spirit_serv
wine = wine_serv

# Create dataframe
chart_data = pd.DataFrame({
    "Alcohol Type": ["Beer", "Spirit", "Wine"],
    "Servings": [beer, spirit, wine]
})

st.subheader("Your Alcohol Consumption Inputs")

st.bar_chart(
    chart_data.set_index("Alcohol Type")
)