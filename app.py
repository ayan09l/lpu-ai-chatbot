import streamlit as st
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load data.txt for custom responses
if os.path.exists("data.txt"):
    with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().strip().lower()
else:
    raw = "lpu is one of the largest universities in india."

sent_tokens = [s.strip() for s in re.split(r'[.!?]\s*', raw) if s.strip()]

# Greetings
GREETING_INPUTS = {"hello", "hi", "greetings", "sup", "hey"}
GREETING_RESPONSES = ["Hi!", "Hello!", "Hey there!", "Hi dear, how can I help?"]

def is_greeting(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return any(w in GREETING_INPUTS for w in words)

def tfidf_response(user_text: str) -> str:
    if not sent_tokens:
        return None
    user_text = user_text.lower().strip()
    corpus = sent_tokens + [user_text]
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    if sims.max() < 0.1:
        return None
    return sent_tokens[sims.argmax()]

# Load AI model (DialoGPT)
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

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot (LPU)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip() != "":
    if user_input.lower() == "bye":
        reply = "Bye! Take care."
    elif is_greeting(user_input):
        reply = random.choice(GREETING_RESPONSES)
    else:
        reply = tfidf_response(user_input) or ai_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

for speaker, msg in st.session_state.history:
    st.markdown(f"{speaker}:** {msg}")
