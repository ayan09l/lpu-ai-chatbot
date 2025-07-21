import streamlit as st
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# ---------------------
# Load LPU Knowledge Base
# ---------------------
def load_knowledge_base():
    if os.path.exists("lpu_data.txt"):
        with open("lpu_data.txt", "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().strip().lower()
    else:
        raw = "LPU is a private university in Punjab, India."
    return [s.strip() for s in re.split(r'[.!?]\s*', raw) if s.strip()]

lpu_sentences = load_knowledge_base()

# ---------------------
# Load General Data
# ---------------------
if os.path.exists("data.txt"):
    with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip().lower()
else:
    raw = "lpu is one of the largest universities in india."

sent_tokens = [s.strip() for s in re.split(r'[.!?]\s*', raw) if s.strip()]

GREETING_INPUTS = {"hello", "hi", "greetings", "sup", "hey"}
GREETING_RESPONSES = ["Hi!", "Hello!", "Hey there!", "Hi dear, how can I help?"]

def is_greeting(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return any(w in GREETING_INPUTS for w in words)

# ---------------------
# TF-IDF based Response
# ---------------------
def tfidf_match(user_text, sentences):
    user_text = user_text.lower().strip()
    corpus = sentences + [user_text]
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    if sims.max() < 0.2:
        return None
    return sentences[sims.argmax()]

def get_response(user_text):
    if is_greeting(user_text):
        return random.choice(GREETING_RESPONSES)

    lpu_answer = tfidf_match(user_text, lpu_sentences)
    if lpu_answer:
        return lpu_answer.capitalize()

    general_answer = tfidf_match(user_text, sent_tokens)
    if general_answer:
        return general_answer.capitalize()

    return ai_response(user_text)

# ---------------------
# DialoGPT Model
# ---------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

def ai_response(user_text: str) -> str:
    new_user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
    if "chat_history_ids" in st.session_state:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# ---------------------
# Voice Response
# ---------------------
def speak(text):
    tts = gTTS(text)
    tts.save("reply.mp3")
    audio_file = open("reply.mp3", "rb")
    st.audio(audio_file, format="audio/mp3")

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Welcome Screen State
if "welcome_done" not in st.session_state:
    st.session_state.welcome_done = False

# ---------------------
# Theme toggle
# ---------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme_choice = st.sidebar.radio("Choose Theme", ["Dark", "Light"])
st.session_state.theme = theme_choice

# Apply Theme
if st.session_state.theme == "Dark":
    st.markdown("""
        <style>
            body, .stApp { background-color: #1e1e1e; color: #f0f0f0; }
            .user-bubble { background: linear-gradient(135deg, #00ff99, #0066ff); color: white; padding: 8px 12px; border-radius: 15px; margin: 5px; max-width: 70%; float: right; clear: both; }
            .bot-bubble { background: linear-gradient(135deg, #333333, #555555); color: #f0f0f0; padding: 8px 12px; border-radius: 15px; margin: 5px; max-width: 70%; float: left; clear: both; }
            .stButton>button, .stDownloadButton>button { background: linear-gradient(90deg, #ff0080, #7928ca); color: white; border-radius: 12px; padding: 6px 15px; border: none; }
            .clearfix { clear: both; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: #ffffff; color: #000000; }
            .user-bubble { background-color: #DCF8C6; padding: 8px 12px; border-radius: 15px; margin: 5px; max-width: 70%; float: right; clear: both; }
            .bot-bubble { background-color: #E6E6E6; padding: 8px 12px; border-radius: 15px; margin: 5px; max-width: 70%; float: left; clear: both; }
            .stButton>button, .stDownloadButton>button { background-color: #007bff; color: white; border-radius: 12px; padding: 6px 15px; border: none; }
            .clearfix { clear: both; }
        </style>
    """, unsafe_allow_html=True)

# ---------------------
# Welcome Screen
# ---------------------
if not st.session_state.welcome_done:
    st.image("lpu_logo.png", width=150)
    st.markdown("<h1 style='text-align:center; color:#ff6600;'>Welcome to Ayush's AI Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Powered by LPU Hackathon Project</p>", unsafe_allow_html=True)
    if st.button("🚀 Start Chat"):
        st.session_state.welcome_done = True
        st.experimental_rerun()
    st.stop()

# ---------------------
# Sidebar Info
# ---------------------
st.sidebar.title("ℹ Chatbot Info")
st.sidebar.markdown("""
*🤖 AI Chatbot*  
- Specialized in *LPU Information*  
- Uses *DialoGPT + TF-IDF*  
- Voice & Text supported  

*Credits:*  
👨‍💻 Developed by Ayush Panigrahi  

*University:*  
Lovely Professional University (LPU)  
B.Tech CSE (AI & ML) Student
""")

# ---------------------
# Chat UI
# ---------------------
st.title("💬 AI Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2)
with col1:
    if st.button("🗑 Clear Chat"):
        st.session_state.history = []
        st.session_state.chat_history_ids = None
        st.rerun()
with col2:
    if st.session_state.history:
        chat_text = "\n".join([f"{speaker}: {msg}" for speaker, msg in st.session_state.history])
        st.download_button("📥 Download Chat", chat_text, file_name="chat_history.txt")

user_input = st.text_input("Type here:", "")

audio_text = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop", just_once=True)
if audio_text and audio_text.strip() != "":
    user_input = audio_text.strip()

if st.button("Send") and user_input.strip() != "":
    reply = get_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))
    speak(reply)

for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div class='user-bubble'>{msg}</div><div class='clearfix'></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}</div><div class='clearfix'></div>", unsafe_allow_html=True)
