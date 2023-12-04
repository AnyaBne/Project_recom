import streamlit as st
import pandas as pd
import requests
import gzip
import shutil
import joblib

# Fonction pour télécharger, décompresser et charger le modèle
@st.cache(allow_output_mutation=True)
def load_model(url):
    # Télécharger le fichier compressé
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("model_recom.pkl.gz", "wb") as f:
            f.write(response.raw.read())

    # Décompresser le fichier
    with gzip.open("model_recom.pkl.gz", "rb") as f_in:
        with open("model_recom.pkl", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Charger le modèle
    return joblib.load("model_recom.pkl")

# URL du fichier modèle compressé sur GitHub
model_url = "model_recom.pkl.gz"
model = load_model(model_url)

# Charger le dataset
data = pd.read_csv('song_dataset.csv')

# Extraire les IDs uniques des utilisateurs
unique_user_ids = data['user'].unique()

# Fonction pour obtenir des recommandations basées sur le modèle
def get_recommendations(user_id, model, n=10):
    listened_songs = data[data['user'] == user_id]['song'].unique()
    all_songs = data[~data['song'].isin(listened_songs)]['song'].unique()
    predictions = []
    for song in all_songs:
        predictions.append((song, model.predict(user_id, song).est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_song_ids = [song for song, _ in predictions[:n]]
    recommended_songs = data[data['song'].isin(recommended_song_ids)][['song', 'title']].drop_duplicates().head(n)
    return recommended_songs

# Fonction pour afficher la page d'accueil
def show_home_page():
    st.title('Welcome to the Music Recommendation Engine')
    user_id = st.text_input('Enter your ID:', key="user_id_input")
    if st.button('Login', key='login_button'):
        if user_id in unique_user_ids:
            st.session_state['page'] = 'recommendation'
            st.session_state['user_id'] = user_id
        else:
            st.error('ID doesn\'t exist. Please try again.')

# Fonction pour afficher la page de recommandation
def show_recommendation_page():
    st.title('Music Recommendation Engine')
    st.write(f"Welcome user {st.session_state.get('user_id', '')}.")

    if 'user_id' in st.session_state and st.session_state['user_id']:
        recommended_songs = get_recommendations(st.session_state['user_id'], model)
        st.subheader('Recommended Songs for you:')
        for _, row in recommended_songs.iterrows():
            st.write(f"{row['song']} - {row['title']}")

    if st.button('Back to Home', key='back_home_button'):
        st.session_state['page'] = 'home'

# Initialisation de l'état de la session
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = ''

# Affichage des pages en fonction de l'état
if st.session_state['page'] == 'home':
    show_home_page()
elif st.session_state['page'] == 'recommendation':
    show_recommendation_page()
