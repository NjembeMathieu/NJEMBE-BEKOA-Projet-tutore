from google import genai
import streamlit as st

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

for m in client.models.list():
    print("MODEL NAME:", m.name)