import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import pickle

# Load the model
model = load_model(r"C:\Users\Gyegn\OneDrive\Desktop\LSTM+RNN\My_LSTM\LSTM_trained_model.keras")


# Load the tokenizer
# with open('tokenizer.pickle','rb') as handle:
#     tokenizer = pickle.load(handle)
tokenizer = Tokenizer()

with open(r"C:\Users\Gyegn\OneDrive\Desktop\LSTM+RNN\My_LSTM\my_venv_lstm\hamlet.txt",'r') as file:
    text = file.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index)+1


# Function to predict next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# Streamlit app
st.title('Predicting next word with LSTM')
input_text = st.text_input('Enter the sequence of the words')
if st.button('Predict Next Word'):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next word: {next_word}')