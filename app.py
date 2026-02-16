import os
import uuid
import asyncio
from typing import List, Optional

import streamlit as st
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from langchain_openai import ChatOpenAI
from PIL import Image

# --- 1. CONFIGURATION ---
# ఇక్కడ మీ OpenAI API కీని నమోదు చేయండి
OS_OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" 
os.environ["OPENAI_API_KEY"] = OS_OPENAI_API_KEY

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 2. CUSTOM MEDICAL TOOLS ---
class SkinScanTool(BaseTool):
    name: str = "SkinScan_Vision_Tool"
    description: str = "చర్మ వ్యాధుల ఫోటోలను విశ్లేషించి మెడికల్ రిపోర్ట్ ఇస్తుంది."

    def _run(self, image_path: str) -> str:
        # డెమో అనాలిసిస్ లాజిక్
        return f"అనాలిసిస్ పూర్తి: '{os.path.basename(image_path)}' చిత్రంలో ఎరుపు రంగు మచ్చలు మరియు వాపు కనిపిస్తున్నాయి. ఇది 'Atopic Dermatitis' లక్షణాలను పోలి ఉంది."

# --- 3. CREWAI AGENTS SETUP ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
vision_tool = SkinScanTool()

dermatologist = Agent(
    role='Dermatology Specialist',
    goal='ఫోటోల ద్వారా చర్మ వ్యాధులను కచ్చితంగా గుర్తించడం.',
    backstory='మీరు చర్మ వ్యాధుల నిపుణులు. మెషిన్ లెర్నింగ్ రిపోర్టులను చదవడంలో మీరు దిట్ట.',
    tools=[vision_tool],
    llm=llm,
    verbose=True
)

cmo = Agent(
    role='Chief Medical Officer',
    goal='తుది మెడికల్ సలహా మరియు జాగ్రత్తలను వివరించడం.',
    backstory='మీరు హాస్పిటల్ బోర్డు హెడ్. రోగికి ఇచ్చే సమాచారం సురక్షితంగా మరియు స్పష్టంగా ఉండేలా చూస్తారు.',
    llm=llm,
    verbose=True
)

# --- 4. FASTAPI BACKEND ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

async def run_medicrew_logic(text: str, file_path: Optional[str] = None):
    # టాస్క్ ల తయారీ
    if file_path:
        task1 = Task(
            description=f"ఈ ఇమేజ్‌ని స్కాన్ చేయండి: {file_path}. యూజర్ అడిగిన ప్రశ్న: {text}",
            expected_output="వివరణాత్మక డెర్మటాలజీ రిపోర్ట్.",
            agent=dermatologist
        )
    else:
        task1 = Task(
            description=f"యూజర్ అడిగిన ఆరోగ్య సమస్యను విశ్లేషించండి: {text}",
            expected_output="ప్రాథమిక విశ్లేషణ.",
            agent=cmo
        )

    task2 = Task(
        description="రిపోర్టును సమీక్షించి రోగికి సులభమైన తెలుగు/ఇంగ్లీష్ భాషలో సలహా ఇవ్వండి.",
        expected_output="ఫైనల్ హెల్త్ అడ్వైజరీ రిపోర్ట్.",
        agent=cmo
    )

    crew = Crew(agents=[dermatologist, cmo], tasks=[task1, task2], process=Process.sequential)
    return str(crew.kickoff())

# --- 5. STREAMLIT GUI (MediCrew AI) ---
st.set_page_config(page_title="MediCrew AI", page_icon="⚕️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #007bff; color: white; }
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 10px; }
    .user-bubble { background-color: #e1f5fe; border-left: 5px solid #03a9f4; }
    .ai-bubble { background-color: #ffffff; border-left: 5px solid #4caf50; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #007bff;'>⚕️ MediCrew AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Autonomous Multi-Agent Medical Diagnosis System</p>", unsafe_allow_html=True)
st.divider()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 అప్‌లోడ్ సెక్షన్")
    uploaded_file = st.file_uploader("చర్మ సమస్య ఫోటోను అప్‌లోడ్ చేయండి", type=['jpg', 'png', 'jpeg'])
    user_input = st.text_area("మీ లక్షణాలను వివరించండి (Symptoms):", placeholder="ఉదా: నాకు రెండు రోజులుగా ఒంటిపై దురదగా ఉంది...")
    submit_btn = st.button("Analyze with MediCrew")

with col2:
    st.subheader("💬 మెడికల్ కన్సల్టేషన్")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # చావట్ హిస్టరీ డిస్ప్లే
    for msg in st.session_state.messages:
        role_class = "user-bubble" if msg["role"] == "user" else "ai-bubble"
        st.markdown(f"<div class='chat-bubble {role_class}'><b>{msg['role'].upper()}:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

    if submit_btn:
        if user_input or uploaded_file:
            # యూజర్ మెసేజ్ సేవ్
            st.session_state.messages.append({"role": "user", "content": user_input if user_input else "Image uploaded for analysis."})
            
            with st.spinner("MediCrew ఏజెంట్లు చర్చిస్తున్నారు..."):
                file_path = None
                if uploaded_file:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # CrewAI రన్ చేయడం
                response = asyncio.run(run_medicrew_logic(user_input, file_path))
                
                # AI మెసేజ్ సేవ్
                st.session_state.messages.append({"role": "ai", "content": response})
                st.rerun()
        else:
            st.warning("దయచేసి సమాచారాన్ని నమోదు చేయండి.")

# Footer
st.markdown("---")
st.caption("⚠️ గమనిక: ఈ నివేదిక కేవలం సమాచారం కోసం మాత్రమే. అత్యవసర పరిస్థితుల్లో వెంటనే డాక్టర్‌ని సంప్రదించండి.")
