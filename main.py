import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Możesz odkomentować i dostosować następujące linie, gdy model będzie gotowy do użycia
# model = joblib.load('model.pkl')
# def predict(input_data):
#     return model.predict([input_data])

# Tworzenie interfejsu użytkownika
st.title('INTELICAR')

# Dodanie pól do wprowadzania danych
with st.form("my_form"):
    rocznik = st.number_input('Rocznik', min_value=1900, max_value=2024, step=1)
    marka = st.text_input('Marka')
    model = st.text_input('Model')
    silnik = st.selectbox('Silnik', options=['Benzyna', 'Diesel', 'Elektryczny', 'Hybrydowy'])
    nadwozie = st.selectbox('Nadwozie', options=['Sedan', 'Hatchback', 'Kombi', 'SUV', 'Coupe', 'Kabriolet'])
    skrzynia = st.selectbox('Skrzynia biegów', options=['Manualna', 'Automatyczna', 'Półautomatyczna'])
    stan = st.slider('Stan', 0, 49, 25)
    przebieg = st.number_input('Przebieg w kilometrach', min_value=0)
    rok_produkcji = st.number_input('Rok produkcji', min_value=1900, max_value=2024, step=1)
    kolor_nadwozia = st.color_picker('Kolor nadwozia')
    kolor_srodka = st.color_picker('Kolor środka')

    # Każde naciśnięcie przycisku 'Wyślij' to wysłanie formularza
    submitted = st.form_submit_button("Wyślij")
    if submitted:
        # Możesz tu użyć funkcji `predict` do przewidywania, przekazując zebrane dane
        # output = predict([rocznik, marka, model, silnik, nadwozie, skrzynia, stan, przebieg, rok_produkcji, kolor_nadwozia, kolor_srodka])
        st.success('Formularz został wysłany.')
        # st.write('Wynik przewidywania:', output)  # Odkomentuj, gdy model będzie dostępny

# Można również użyć poniższego kodu, jeśli chcesz wykonać przewidywanie poza formularzem
# if st.button('Przewiduj'):
#     if data_input:  # Sprawdź czy pole tekstowe nie jest puste
#         output = predict(data_input)
#         st.success(f'Przewidywanie modelu: {output}')
#     else:
#         st.error('Proszę wprowadzić dane.')
