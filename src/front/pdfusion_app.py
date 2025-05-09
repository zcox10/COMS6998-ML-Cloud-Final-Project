import streamlit as st
import requests
from PIL import Image
import base64

st.set_page_config(
    page_title="PDFusion", page_icon="📄", layout="centered", initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    :root {
        --primary-purple: #7A52C7;
        --secondary-white: #FFFFFF;
        --accent-purple: #A07BEF;
        --soft-background: #F5F3FA;
        --button-hover: #5E3BA2;
    }
    
    .main {
        background-color: var(--soft-background);
        padding: 3rem 2rem 2rem 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .block-container {
        max-width: 80%;
        margin: 1.5rem auto 0 auto;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-top: 1.5rem;

    }
    
    .description {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    .input-label {
        font-size: 1.2rem !important;
        font-weight: 600;
        color: var(--primary-purple);
        margin-bottom: 0.5rem;
    }
    
    .stTextInput div[data-baseweb="input"] {
        border-radius: 8px;
        border: 2px solid var(--primary-purple);
    }
    
    .stTextInput div[data-baseweb="input"]:focus-within {
        border-color: var(--accent-purple);
        box-shadow: 0 0 0 2px rgba(122, 82, 199, 0.2);
    }
    
    .stButton button {
        background-color: var(--accent-purple) !important;
        color: var(--secondary-white);
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: var(--button-hover) !important;
        box-shadow: 0 4px 8px rgba(94, 59, 162, 0.3);
        transform: translateY(-2px);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    .output-container {
        background-color: var(--secondary-white);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid var(--primary-purple);
    }
    
    .output-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-purple);
        margin-bottom: 1rem;
    }
    
    .stTextArea textarea {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        background-color: var(--soft-background);
    }
    
    .stAlert {
        border-radius: 8px;
    }
    
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #666;
    }
    
    .heart {
        color: var(--primary-purple);
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    [data-testid="stImage"] {
        text-align: center;
        display: block;
        margin: 0 auto;
    }
    
    .stTextArea > div {
        padding-top: 0 !important;
    }
    
    .stTextArea > div > div {
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    
    .stTextArea label {
        display: none !important;
    }
    
    .css-1544g2n {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    .css-91z34k {
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image("pdfusion.png", width=350)
    except:
        st.markdown(
            '<div class="logo-container"><h1 style="color: #7A52C7; font-size: 3.5rem; font-weight: 700;">📄 PDFusion</h1></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <p class="description">
    A powerful tool to extract and present useful information from research papers, created with Docling & Advanced Language Models (Qwen 3, Google Gemini).
    </p>
    """,
    unsafe_allow_html=True,
)

with st.form(key="paper_form"):
    st.markdown('<p class="input-label">Enter arXiv paper ID or URL:</p>', unsafe_allow_html=True)

    user_input = st.text_input(
        label="",
        placeholder="e.g., 2103.13630 or https://arxiv.org/abs/2103.13630",
        label_visibility="collapsed",
    )

    submit_col1, submit_col2, submit_col3 = st.columns([4, 2, 4])
    with submit_col2:
        submit_button = st.form_submit_button(label="Process Paper")

if submit_button:
    if user_input.strip() != "":
        if "arxiv.org/abs/" in user_input:
            user_input = user_input.split("arxiv.org/abs/")[1].split()[0]

        with st.spinner("Processing paper - This may take a moment"):
            try:
                url = "http://arxiv-summarization-api-service:80/summarize"
                response = requests.post(url, json={"entry_id": user_input}, timeout=60)
                print(response)

                if response.status_code == 200:
                    result_text = response.json().get("response", "")

                    st.markdown(
                        f'<p class="output-header">📑 Paper Details</p>', unsafe_allow_html=True
                    )

                    st.text_area(
                        label="", value=result_text, height=500, label_visibility="collapsed"
                    )

                else:
                    st.error(
                        f"Request failed: {response.status_code}. Paper might not be found or server error occurred."
                    )

            except requests.exceptions.Timeout:
                st.error("Request timed out. The server took too long to respond.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter an arXiv paper ID or URL")

st.markdown(
    """
    <div class="footer">
        <p>Crafted with <span class="heart">❤</span> by Zach, Nirav, Xincheng & Jian</p>
    </div>
    """,
    unsafe_allow_html=True,
)
