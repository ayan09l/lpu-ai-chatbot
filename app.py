import os
import re
import random
import streamlit as st
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# ---------------------------
# FIX: Media File Storage
# ---------------------------
if not os.path.exists("/tmp/streamlit"):
    os.makedirs("/tmp/streamlit")
os.environ["STREAMLIT_MEDIA_FILE_STORAGE"] = "/tmp/streamlit"

# ---------------------------
# Load Knowledge Base
# ---------------------------
def load_knowledge_base(file_name, default_text):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().strip().lower()
    else:
        raw = default_text
    return [s.strip() for s in re.split(r'[.!?]\s*', raw) if s.strip()]

lpu_sentences = load_knowledge_base("lpu_data.txt", "LPU is a private university in Punjab, India.")
general_sentences = load_knowledge_base("data.txt", "LPU is one of the largest universities in India.")

GREETING_INPUTS = {"hello", "hi", "hey", "greetings"}
GREETING_RESPONSES = ["Hello!", "Hi there!", "Hey dear!", "Hi, how can I help you?"]

# ---------------------------
# Helper Functions
# ---------------------------
def is_greeting(user_text):
    return any(word in GREETING_INPUTS for word in user_text.lower().split())

def tfidf_match(user_text, sentences):
    vec = TfidfVectorizer(stop_words="english")
    all_sentences = sentences + [user_text]
    tfidf = vec.fit_transform(all_sentences)
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
    general_answer = tfidf_match(user_text, general_sentences)
    if general_answer:
