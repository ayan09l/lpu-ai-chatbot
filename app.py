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
    if sims.max() < 0.2:  # threshold for relevance
        return None
    return sentences[sims.argmax()]

def get_response(user_text):
    # Priority 1: Greeting
    if is_greeting(user_text):
        return random.choice(GREETING_RESPONSES)

    # Priority 2: LPU Knowledge
    lpu_answer = tfidf_match(user_text, lpu_sentences)
    if lpu_answer:
        return lpu_answer.capitalize()

    # Priority 3: General Data
    general_answer = tfidf_match(user_text, sent_tokens)
    if general_answer:
        return general_answer.capitalize()

    # Priority 4: AI Model
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

# --- Sidebar ---
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

st.markdown("""
    <style>
        .user-bubble {
            background-color: #DCF8C6;
            padding: 8px 12px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #E6E6E6;
            padding: 8px 12px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            float: left;
            clear: both;
        }
        .clearfix { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Chatbot (Voice + Text)")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# --- Buttons Row ---
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

# User input
user_input = st.text_input("Type here:", "")

# Voice input
audio_text = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop", just_once=True)
if audio_text and audio_text.strip() != "":
    user_input = audio_text.strip()

# Send Button
if st.button("Send") and user_input.strip() != "":
    reply = get_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))
    speak(reply)

# Display chat bubbles
for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div class='user-bubble'>{msg}</div><div class='clearfix'></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}</div><div class='clearfix'></div>", unsafe_allow_html=True)
