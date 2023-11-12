import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from nltk.tokenize import word_tokenize

# Load dictionaries
file = open("idx_to_word_data.txt", 'rb')
idx_to_word = pickle.load(file)
file.close()

file = open("word_to_idx_data.txt", 'rb')
word_to_idx = pickle.load(file)
file.close()

max_sequence_len = 123

model = load_model('model.h5')

def predict_next_word(model, text, max_sequence_len, word_to_idx, idx_to_word):
    """
    Predict the next word based on the input text.

    Args:
    - model (tf.keras.Model): Trained model for prediction.
    - text (str): Input string.
    - max_sequence_len (int): Maximum length of input sequences.
    - word_to_index (dict): Mapping from words to their respective indices.
    - index_to_word (dict): Mapping from indices to their respective words.

    Returns:
    - str: Predicted word.
    """

    # Tokenize the input string
    token_list = [word_to_idx[word] for word in word_tokenize(text) if word in word_to_idx]

    # Pad the token sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Predict the token of the next word
    predicted_idx = np.argmax(model.predict(token_list), axis=-1)

    # Convert the token back to a word
    predicted_word = idx_to_word.get(predicted_idx[0], '')

    return predicted_word

with st.sidebar:
    st.markdown(
        "<h1>About Our Model</h1>",
        unsafe_allow_html=True,
    )
    "Natural Language Processing model that predicts the next 'n' words, specified by the user, given a user input."
    st.markdown(
        "<h3>Michigan Data Science Team<sup style='font-size:.8em;'>©</sup></h3>",
        unsafe_allow_html=True,
    )
    "All Rights Reserved"

st.title("➱ Next Word Predictor") 
st.divider()
number = st.slider("Pick a number of words to predict", 0, 25)
st.divider()
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What sequence would you like to predict"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

count = 0

if prompt := st.chat_input(key = count):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = predict_next_word(model, prompt, max_sequence_len, word_to_idx, idx_to_word)
    #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #msg = response.choices[0].message
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(prompt + " " + response)
    count += 1