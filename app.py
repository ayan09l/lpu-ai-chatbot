import streamlit as st
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import os
from streamlit_mic_recorder import speech_to_text

# Load your data.txt for custom responses
if os.path.exists("data.txt"):
    with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip().lower()
else:
    raw = ""

sent_tokens = [s.strip() for s in re.split(r'[.!?]\s*', raw) if s.strip()]

# Greetings
GREETING_INPUTS = {"hello", "hi", "greetings", "sup", "hey"}
GREETING_RESPONSES = ["Hi!", "Hello!", "Hey there!", "Hi dear, how can I help?"]

def is_greeting(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return any(w in GREETING_INPUTS for w in words)

def tfidf_response(user_text: str) -> str:
    if not sent_tokens:
        return "I don't have custom answers yet."
    user_text = user_text.lower().strip()
    corpus = sent_tokens + [user_text]
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    if sims.max() < 0.1:
        return None
    return sent_tokens[sims.argmax()]

# Load AI model (DialoGPT)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def ai_response(user_text: str) -> str:
    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if "chat_history_ids" in st.session_state else new_user_input_ids
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

def speak_text(text: str):
    tts = gTTS(text=text, lang='en')
    tts.save("bot_reply.mp3")
    audio_file = open("bot_reply.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")
    os.remove("bot_reply.mp3")

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot (LPU)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

# Voice Input
st.write("🎙 Speak your query:")
voice_input = speech_to_text(language='en', key='voice')
if voice_input and voice_input.strip() != "":
    user_input = voice_input
    st.write(f"*Voice Input:* {voice_input}")

if st.button("Send") and user_input.strip() != "":
    if user_input.lower() == "bye":
        reply = "Bye! Take care."
    elif is_greeting(user_input):
        reply = random.choice(GREETING_RESPONSES)
    else:
        reply = tfidf_response(user_input) or ai_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))
    speak_text(reply)

for speaker, msg in st.session_state.history:
    st.markdown(f"{speaker}:** {msg}")