import streamlit as st
import os
import glob
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from requests.auth import HTTPBasicAuth
from urllib.parse import urlsplit, urlunsplit

st.set_page_config(page_title="InnoNexus AI Tool", layout="wide")
# --- Remote Ollama HTTP client (replaces the Python ollama module) ---if st.button("Generate statement 📝", key=K("generate_btn")):

# Normalize base URL so endpoints are appended only once
def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    u = urlsplit(url if "://" in url else f"https://{url.lstrip('/')}")
    path = (u.path or "").rstrip("/")
    for bad in ("/api/chat", "/api/generate", "/api"):
        if path.endswith(bad):
            path = path[: -len(bad)]
            break
    path = path.rstrip("/")
    return urlunsplit((u.scheme, u.netloc, path, "", ""))

class RemoteOllama:
    def __init__(self, base_url, auth=None, force_host=False):
        self.base_url = normalize_base_url(base_url)
        self.auth = auth
        self.force_host = force_host

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.force_host:
            h["Host"] = "localhost:11434"
        return h

    def chat(self, model, messages, stream=False, options=None, timeout=120):
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": bool(stream)}
        if options:
            payload["options"] = options
        r = requests.post(url, json=payload, headers=self._headers(),
                          timeout=timeout, auth=self.auth)
        r.raise_for_status()
        return r.json()  # {"message":{"content": "..."}}

# --- Sidebar (single source of truth) ---
st.sidebar.header("InnoNexus")
st.sidebar.subheader("Ollama AI Configuration")
desired_model = st.sidebar.text_input("Enter your Ollama Model Name", value="llama3.1:8b", key="cfg_model_name")
ollama_api_base_raw = st.sidebar.text_input("Ollama API Base URL", value="https://YOUR_TUNNEL.ngrok-free.app", key="cfg_api_base")
with st.sidebar.expander("Advanced (Auth)"):
    auth_user = st.text_input("Basic Auth Username", value="", key="cfg_auth_user")
    auth_pass = st.text_input("Basic Auth Password", value="", type="password", key="cfg_auth_pass")
    force_host = st.checkbox("Force Host header (use only if needed)", value=False, key="cfg_force_host")

# --- Session-scoped client (no globals) ---
def get_ollama_client():
    auth = HTTPBasicAuth(auth_user, auth_pass) if auth_user and auth_pass else None
    cleaned = normalize_base_url(ollama_api_base_raw)
    key = (cleaned, auth_user, bool(auth_pass), force_host)
    if "ollama_client_key" not in st.session_state or st.session_state["ollama_client_key"] != key:
        st.session_state["ollama_client"] = RemoteOllama(cleaned, auth=auth, force_host=force_host)
        st.session_state["ollama_client_key"] = key
    return st.session_state["ollama_client"]

# Default model into session for easy access
st.session_state.setdefault("desired_model", None)
if desired_model:
    st.session_state["desired_model"] = desired_model

def chat_text(prompt: str, model: str = None, options=None, timeout=120) -> str:
    """Session-safe wrapper around Ollama /api/chat (non-streaming)."""
    client = get_ollama_client()
    use_model = model or st.session_state.get("desired_model") or "llama3.1:8b"
    payload_msgs = [{"role": "user", "content": prompt.strip()}]
    resp = client.chat(model=use_model, messages=payload_msgs, stream=False,
                       options=options, timeout=timeout)
    return resp["message"]["content"].strip()





# -----------------------------------------------------------------------------
# Caching for Performance
# -----------------------------------------------------------------------------

@st.cache_resource
def load_sentence_model():
    """Loads the sentence transformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_patent_data(uploaded_file):
    """Loads and caches the patent data from the uploaded file."""
    try:
        # Check the file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # --- FIX: Tell pandas to use the second row (index 1) as the header ---
            df = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
            
        # Make column names lowercase and strip whitespace for consistency
        df.columns = [str(col).lower().strip() for col in df.columns]
        return df

    except UnicodeDecodeError:
        # If UTF-8 fails for a CSV, try a more lenient encoding
        st.warning("UTF-8 decoding failed. Trying a different encoding (latin1).")
        uploaded_file.seek(0) # Reset file pointer
        df = pd.read_csv(uploaded_file, encoding='latin1')
        df.columns = [str(col).lower().strip() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

@st.cache_data
def generate_patent_embeddings(_df, _model):
    """
    Generates and caches embeddings for patent abstracts.
    The '_model' argument has a leading underscore to tell Streamlit not to hash it.
    """
    if 'abstract' in _df.columns:
        # Ensure all abstracts are strings, replacing non-string data with an empty string
        abstracts = _df['abstract'].astype(str).fillna('').tolist()
        return _model.encode(abstracts, convert_to_tensor=True)
    return None

# -----------------------------------------------------------------------------
# TRIZ Data
# -----------------------------------------------------------------------------

TRIZ_PARAMETERS = [
    "1. Weight of moving object", "2. Weight of stationary object", "3. Length of moving object",
    "4. Length of stationary object", "5. Area of moving object", "6. Area of stationary object",
    "7. Volume of moving object", "8. Volume of stationary object", "9. Speed", "10. Force",
    "11. Stress or pressure", "12. Shape", "13. Stability of the object's composition",
    "14. Strength", "15. Duration of action by a moving object", "16. Duration of action by a stationary object",
    "17. Temperature", "18. Illumination intensity", "19. Use of energy by moving object",
    "20. Use of energy by stationary object", "21. Power", "22. Loss of Energy", "23. Loss of substance",
    "24. Loss of Information", "25. Loss of Time", "26. Quantity of substance", "27. Reliability",
    "28. Measurement accuracy", "29. Manufacturing precision", "30. Object-affected harmful factors",
    "31. Object-generated harmful factors", "32. Ease of manufacture", "33. Ease of operation",
    "34. Ease of repair", "35. Adaptability or versatility", "36. Device complexity",
    "37. Difficulty of detecting and measuring", "38. Extent of automation", "39. Productivity"
]

TRIZ_PRINCIPLES = {
    1: ("Segmentation", "Divide an object into independent parts."),
    2: ("Taking out", "Separate an interfering part or property from an object."),
    3: ("Local quality", "Transition from a homogeneous to a heterogeneous structure."),
    4: ("Asymmetry", "Change the shape from symmetrical to asymmetrical."),
    5: ("Merging", "Bring closer together similar objects or operations in space or time."),
    6: ("Universality", "Make a part or object perform multiple functions."),
    7: ("Nested doll", "Place one object inside another."),
    8: ("Anti-weight", "Compensate for the weight of an object by merging it with other objects that provide lift."),
    9: ("Preliminary anti-action", "Perform an action with opposite effects in advance."),
    10: ("Preliminary action", "Perform, before it is needed, the required change of an object."),
    11: ("Beforehand cushioning", "Prepare emergency means beforehand."),
    12: ("Equipotentiality", "Limit position changes in a potential field."),
    13: ("The other way round", "Invert the action(s) used to solve the problem."),
    14: ("Spheroidality - Curvature", "Use curvilinear parts, surfaces, or forms."),
    15: ("Dynamics", "Allow characteristics to change to be optimal."),
    16: ("Partial or excessive actions", "Use 'slightly less' or 'slightly more' of the same method."),
    17: ("Another dimension", "Move an object in two- or three-dimensional space."),
    18: ("Mechanical vibration", "Cause an object to oscillate or vibrate."),
    19: ("Periodic action", "Use periodic or pulsating actions instead of continuous action."),
    20: ("Continuity of useful action", "Carry on work continuously."),
    21: ("Skipping", "Conduct a process at high speed."),
    22: ("Blessing in disguise", "Use harmful factors to achieve a positive effect."),
    23: ("Feedback", "Introduce feedback to improve a process or action."),
    24: ("Intermediary", "Use an intermediary carrier article or process."),
    25: ("Self-service", "Make an object serve itself."),
    26: ("Copying", "Use simpler and inexpensive copies."),
    27: ("Cheap short-living objects", "Replace an expensive object with a multitude of inexpensive ones."),
    28: ("Mechanics substitution", "Replace a mechanical means with a sensory one."),
    29: ("Pneumatics and hydraulics", "Use gas and liquid parts of an object instead of solid parts."),
    30: ("Flexible shells and thin films", "Use flexible shells and thin films."),
    31: ("Porous materials", "Make an object porous or use porous elements."),
    32: ("Color changes", "Change the color of an object or its external environment."),
    33: ("Homogeneity", "Make objects interacting with a given object of the same material."),
    34: ("Discarding and recovering", "Make portions of an object that have fulfilled their functions go away."),
    35: ("Parameter changes", "Change an object's physical state, concentration, or flexibility."),
    36: ("Phase transitions", "Use phenomena occurring during phase transitions."),
    37: ("Thermal expansion", "Use thermal expansion or contraction of materials."),
    38: ("Strong oxidants", "Replace common air with enriched air or oxygen."),
    39: ("Inert atmosphere", "Replace a normal environment with an inert one."),
    40: ("Composite materials", "Change from uniform to composite materials."),
}

# The 39x39 Contradiction Matrix (Updated Classical Version)
CONTRADICTION_MATRIX =[
[[], [], [15, 8, 29, 34], [], [29, 17, 38, 34], [], [29, 2, 40, 28], [], [2, 8, 15, 38], [8, 10, 18, 37], [10, 36, 37, 40], [10, 14, 35, 40], [1, 35, 19, 39], [28, 27, 18, 40], [5, 34, 31, 35], [], [6, 29, 4, 38], [19, 1, 32], [35, 12, 34, 31], [], [12, 36, 18, 31], [6, 2, 34, 19], [5, 35, 3, 31], [10, 24, 35], [10, 35, 20, 28], [3, 26, 18, 31], [1, 3, 11, 27], [28, 27, 35, 26], [28, 35, 26, 18], [22, 21, 18, 27], [22, 35, 31, 39], [27, 28, 1, 36], [35, 3, 2, 24], [2, 27, 28, 11], [29, 5, 15, 8], [26, 30, 36, 34], [28, 29, 26, 32], [26, 3518, 19], [35, 3, 24, 37]], [[], [], [], [10, 1, 29, 35], [], [35, 30, 13, 2], [], [5, 35, 14, 2], [], [8, 10, 19, 35], [13, 29, 10, 18], [13, 10, 29, 14], [26, 39, 1, 40], [28, 2, 10, 27], [], [2, 27, 19, 6], [28, 19, 32, 22], [19, 32, 35], [], [18, 19, 28, 1], [15, 19, 18, 22], [18, 19, 28, 15], [5, 8, 13, 30], [10, 15, 35], [10, 20, 35, 26], [19, 6, 18, 26], [10, 28, 8, 3], [18, 26, 28], [10, 1, 35, 17], [2, 19, 22, 37], [35, 22, 1, 39], [28, 1, 9], [6, 13, 1, 32], [2, 27, 28, 11], [19, 15, 29], [1, 10, 26, 39], [25, 28, 17, 15], [2, 26, 35], [1, 28, 15, 35]], [[8, 15, 29, 34], [], [], [], [15, 17, 4], [], [7, 17, 4, 35], [], [13, 4, 8], [17, 10, 4], [1, 8, 35], [1, 8, 10, 29], [1, 8, 15, 34], [8, 35, 29, 34], [19], [], [10, 15, 19], [32], [8, 35, 24], [], [1, 35], [7, 2, 35, 39], [4, 29, 23, 10], [1, 24], [15, 2, 29], [29, 35], [10, 14, 29, 40], [28, 32, 4], [10, 28, 29, 37], [1, 15, 17, 24], [17, 15], [1, 29, 17], [15, 29, 35, 4], [1, 28, 10], [14, 15, 1, 16], [1, 19, 26, 24], [35, 1, 26, 24], [17, 24, 26, 16], [14, 4, 28, 29]], [[], [35, 28, 40, 29], [], [], [], [17, 7, 10, 40], [], [35, 8, 2, 14], [], [28, 10], [1, 14, 35], [13, 14, 15, 7], [39, 37, 35], [15, 14, 28, 26], [], [1, 10, 35], [3, 35, 38, 18], [3, 25], [], [], [12, 8], [6, 28], [10, 28, 24, 35], [24, 26], [30, 29, 14], [], [15, 29, 28], [32, 28, 3], [2, 32, 10], [1, 18], [], [15, 17, 27], [2, 25], [3], [1, 35], [1, 26], [26], [], [30, 14, 7, 26]], [[2, 17, 29, 4], [], [14, 15, 18, 4], [], [], [], [7, 14, 17, 4], [], [29, 30, 4, 34], [19, 30, 35, 2], [10, 15, 36, 28], [5, 34, 29, 4], [11, 2, 13, 39], [3, 15, 40, 14], [6, 3], [], [2, 15, 16], [15, 32, 19, 13], [19, 32], [], [19, 10, 32, 18], [15, 17, 30, 26], [10, 35, 2, 39], [30, 26], [26, 4], [29, 30, 6, 13], [29, 9], [26, 28, 32, 3], [2, 32], [22, 33, 28, 1], [17, 2, 18, 39], [13, 1, 26, 24], [15, 17, 13, 16], [15, 13, 10, 1], [15, 30], [14, 1, 13], [2, 36, 26, 18], [14, 30, 28, 23], [10, 26, 34, 2]], [[], [30, 2, 14, 18], [], [26, 7, 9, 39], [], [], [], [], [], [1, 18, 35, 36], [10, 15, 36, 37], [], [2, 38], [40], [], [2, 10, 19, 30], [35, 39, 38], [], [], [], [17, 32], [17, 7, 30], [10, 14, 18, 39], [30, 16], [10, 35, 4, 18], [2, 18, 40, 4], [32, 35, 40, 4], [26, 28, 32, 3], [2, 29, 18, 36], [27, 2, 39, 35], [22, 1, 40], [40, 16], [16, 4], [16], [15, 16], [1, 18, 36], [2, 35, 30, 18], [23], [10, 15, 17, 7]], [[2, 26, 29, 40], [], [1, 7, 4, 35], [], [1, 7, 4, 17], [], [], [], [29, 4, 38, 34], [15, 35, 36, 37], [6, 35, 36, 37], [1, 15, 29, 4], [28, 10, 1, 39], [9, 14, 15, 7], [6, 35, 4], [], [34, 39, 10, 18], [2, 13, 10], [35], [], [35, 6, 13, 18], [7, 15, 13, 16], [36, 39, 34, 10], [2, 22], [2, 6, 34, 10], [29, 30, 7], [14, 1, 40, 11], [25, 26, 28], [25, 28, 2, 16], [22, 21, 27, 35], [17, 2, 40, 1], [29, 1, 40], [15, 13, 30, 12], [10], [15, 29], [26, 1], [29, 26, 4], [35, 34, 16, 24], [10, 6, 2, 34]], [[], [35, 10, 19, 14], [19, 14], [35, 8, 2, 14], [], [], [], [], [], [2, 18, 37], [24, 35], [7, 2, 35], [34, 28, 35, 40], [9, 14, 17, 15], [], [35, 34, 38], [35, 6, 4], [], [], [], [30, 6], [], [10, 39, 35, 34], [], [35, 16, 3218], [35, 3], [2, 35, 16], [], [35, 10, 25], [34, 39, 19, 27], [30, 18, 35, 4], [35], [], [1], [], [1, 31], [2, 17, 26], [], [35, 37, 10, 2]], [[2, 28, 13, 38], [], [13, 14, 8], [], [29, 30, 34], [], [7, 29, 34], [], [], [13, 28, 15, 19], [6, 18, 38, 40], [35, 15, 18, 34], [28, 33, 1, 18], [8, 3, 26, 14], [3, 19, 35, 5], [], [28, 30, 36, 2], [10, 13, 19], [8, 15, 35, 38], [], [19, 35, 38, 2], [14, 20, 19, 35], [10, 13, 28, 38], [13, 26], [], [10, 19, 29, 38], [11, 35, 27, 28], [28, 32, 1, 24], [10, 28, 32, 25], [1, 28, 35, 23], [2, 24, 35, 21], [35, 13, 8, 1], [32, 28, 13, 12], [34, 2, 28, 27], [15, 10, 26], [10, 28, 4, 34], [3, 34, 27, 16], [10, 18], []], [[8, 1, 37, 18], [18, 13, 1, 28], [17, 19, 9, 36], [28, 10], [19, 10, 15], [1, 18, 36, 37], [15, 9, 12, 37], [2, 36, 18, 37], [13, 28, 15, 12], [], [18, 21, 11], [10, 35, 40, 34], [35, 10, 21], [35, 10, 14, 27], [19, 2], [], [35, 10, 21], [], [19, 17, 10], [1, 16, 36, 37], [19, 35, 18, 37], [14, 15], [8, 35, 40, 5], [], [10, 37, 36], [14, 29, 18, 36], [3, 35, 13, 21], [35, 10, 23, 24], [28, 29, 37, 36], [1, 35, 40, 18], [13, 3, 36, 24], [15, 37, 18, 1], [1, 28, 3, 25], [15, 1, 11], [15, 17, 18, 20], [26, 35, 10, 18], [36, 37, 10, 19], [2, 35], [3, 28, 35, 37]], [[10, 36, 37, 40], [13, 29, 10, 18], [35, 10, 36], [35, 1, 14, 16], [10, 15, 36, 28], [10, 15, 36, 37], [6, 35, 10], [35, 24], [6, 35, 36], [36, 35, 21], [], [35, 4, 15, 10], [35, 33, 2, 40], [9, 18, 3, 40], [19, 3, 27], [], [35, 39, 19, 2], [], [14, 24, 10, 37], [], [10, 35, 14], [2, 36, 25], [10, 36, 3, 37], [], [37, 36, 4], [10, 14, 36], [10, 13, 19, 35], [6, 28, 25], [3, 35], [22, 2, 37], [2, 33, 27, 18], [1, 35, 16], [11], [2], [35], [19, 1, 35], [2, 36, 37], [35, 24], [10, 14, 35, 37]], [[8, 10, 29, 40], [15, 10, 26, 3], [29, 34, 5, 4], [13, 14, 10, 7], [5, 34, 4, 10], [], [14, 4, 15, 22], [7, 2, 35], [35, 15, 34, 18], [35, 10, 37, 40], [34, 15, 10, 14], [], [33, 1, 18, 4], [30, 14, 10, 40], [14, 26, 9, 25], [], [22, 14, 19, 32], [13, 15, 32], [2, 6, 34, 14], [], [4, 6, 2], [14], [35, 29, 3, 5], [], [14, 10, 34, 17], [36, 22], [10, 40, 16], [28, 32, 1], [32, 30, 40], [22, 1, 2, 35], [35, 1], [1, 32, 17, 28], [32, 15, 26], [2, 13, 1], [1, 15, 29], [16, 29, 1, 28], [15, 13, 39], [15, 1, 32], [17, 26, 34, 10]], [[21, 35, 2, 39], [26, 39, 1, 40], [13, 15, 1, 28], [37], [2, 11, 13], [39], [28, 10, 19, 39], [34, 28, 35, 40], [33, 15, 28, 18], [10, 35, 21, 16], [2, 35, 40], [22, 1, 18, 4], [], [17, 9, 15], [13, 27, 10, 35], [39, 3, 35, 23], [35, 1, 32], [32, 3, 27, 16], [13, 19], [27, 4, 29, 18], [32, 35, 27, 31], [14, 2, 39, 6], [2, 14, 30, 40], [], [35, 27], [15, 32, 35], [], [13], [18], [35, 24, 30, 18], [35, 40, 27, 39], [35, 19], [32, 35, 30], [2, 35, 10, 16], [35, 30, 34, 2], [2, 35, 22, 26], [35, 22, 39, 23], [1, 8, 35], [23, 35, 40, 3]], [[1, 8, 40, 15], [40, 26, 27, 1], [1, 15, 8, 35], [15, 14, 28, 26], [3, 34, 40, 29], [9, 40, 28], [10, 15, 14, 7], [9, 14, 17, 15], [8, 13, 26, 14], [10, 18, 3, 14], [10, 3, 18, 40], [10, 30, 35, 40], [13, 17, 35], [], [27, 3, 26], [], [30, 10, 40], [35, 19], [19, 35, 10], [35], [10, 26, 35, 28], [35], [35, 28, 31, 40], [], [29, 3, 28, 10], [29, 10, 27], [11, 3], [3, 27, 16], [3, 27], [18, 35, 37, 1], [15, 35, 22, 2], [11, 3, 10, 32], [32, 40, 25, 2], [27, 11, 3], [15, 3, 32], [2, 13, 25, 28], [27, 3, 15, 40], [15], [29, 35, 10, 14]], [[19, 5, 34, 31], [], [2, 19, 9], [], [3, 17, 19], [], [10, 2, 19, 30], [], [3, 35, 5], [19, 2, 16], [19, 3, 27], [14, 26, 28, 25], [13, 3, 35], [27, 3, 10], [], [], [19, 35, 39], [2, 19, 4, 35], [28, 6, 35, 18], [], [19, 10, 35, 38], [], [28, 27, 3, 18], [10], [20, 10, 28, 18], [3, 35, 10, 40], [11, 2, 13], [3], [3, 27, 16, 40], [22, 15, 33, 28], [21, 39, 16, 22], [27, 1, 4], [12, 27], [29, 10, 27], [1, 35, 13], [10, 4, 29, 15], [19, 29, 39, 35], [6, 10], [35, 17, 14, 19]], [[], [6, 27, 19, 16], [], [1, 40, 35], [], [], [], [35, 34, 38], [], [], [], [], [39, 3, 35, 23], [], [], [], [19, 18, 36, 40], [], [], [], [16], [], [27, 16, 18, 38], [10], [28, 20, 10, 16], [3, 35, 31], [34, 27, 6, 40], [10, 26, 24], [], [17, 1, 40, 33], [22], [35, 10], [1], [1], [2], [], [25, 34, 6, 35], [1], [20, 10, 16, 38]], [[36, 22, 6, 38], [22, 35, 32], [15, 19, 9], [15, 19, 9], [3, 35, 39, 18], [35, 38], [34, 39, 40, 18], [35, 6, 4], [2, 28, 36, 30], [35, 10, 3, 21], [35, 39, 19, 2], [14, 22, 19, 32], [1, 35, 32], [10, 30, 22, 40], [19, 13, 39], [19, 18, 36, 40], [], [32, 30, 21, 16], [19, 15, 3, 17], [], [2, 14, 17, 25], [21, 17, 35, 38], [21, 36, 29, 31], [], [35, 28, 21, 18], [3, 17, 30, 39], [19, 35, 3, 10], [32, 19, 24], [24], [22, 33, 35, 2], [22, 35, 2, 24], [26, 27], [26, 27], [4, 10, 16], [2, 18, 27], [2, 17, 16], [3, 27, 35, 31], [26, 2, 19, 16], [15, 28, 35]], [[19, 1, 32], [2, 35, 32], [19, 32, 16], [], [19, 32, 26], [], [2, 13, 10], [], [10, 13, 19], [26, 19, 6], [], [32, 30], [32, 3, 27], [35, 19], [2, 19, 6], [], [32, 35, 19], [], [32, 1, 19], [32, 35, 1, 15], [32], [13, 16, 1, 6], [13, 1], [1, 6], [19, 1, 26, 17], [1, 19], [], [11, 15, 32], [3, 32], [15, 19], [35, 19, 32, 39], [19, 35, 28, 26], [28, 26, 19], [15, 17, 13, 16], [15, 1, 19], [6, 32, 13], [32, 15], [2, 26, 10], [2, 25, 16]], [[12, 18, 28, 31], [], [12, 28], [], [15, 19, 25], [], [35, 13, 18], [], [8, 35, 35], [16, 26, 21, 2], [23, 14, 25], [12, 2, 29], [19, 13, 17, 24], [5, 19, 9, 35], [28, 35, 6, 18], [], [19, 24, 3, 14], [2, 15, 19], [], [], [6, 19, 37, 18], [12, 22, 15, 24], [35, 24, 18, 5], [], [35, 38, 19, 18], [34, 23, 16, 18], [19, 21, 11, 27], [3, 1, 32], [], [1, 35, 6, 27], [2, 35, 6], [28, 26, 30], [19, 35], [1, 15, 17, 28], [15, 17, 13, 16], [2, 29, 27, 28], [35, 38], [32, 2], [12, 28, 35]], [[], [19, 9, 6, 27], [], [], [], [], [], [], [], [36, 37], [], [], [27, 4, 29, 18], [35], [], [], [], [19, 2, 35, 32], [], [], [], [], [28, 27, 18, 31], [], [], [3, 35, 31], [10, 36, 23], [], [], [10, 2, 22, 37], [19, 22, 18], [1, 4], [], [], [], [], [19, 35, 16, 25], [], [1, 6]], [[8, 36, 38, 31], [19, 26, 17, 27], [1, 10, 35, 37], [], [19, 38], [17, 32, 13, 38], [35, 6, 38], [30, 6, 25], [15, 35, 2], [26, 2, 36, 35], [22, 10, 35], [29, 14, 2, 40], [35, 32, 15, 31], [26, 10, 28], [19, 35, 10, 38], [16], [2, 14, 17, 25], [16, 6, 19], [16, 6, 19, 37], [], [], [10, 35, 38], [28, 27, 18, 38], [10, 19], [35, 20, 10, 6], [4, 34, 19], [19, 24, 26, 31], [32, 15, 2], [32, 2], [19, 22, 31, 2], [2, 35, 18], [26, 10, 34], [26, 35, 10], [35, 2, 10, 34], [19, 17, 34], [20, 19, 30, 34], [19, 35, 16], [28, 2, 17], [28, 35, 34]], [[15, 6, 19, 28], [19, 6, 18, 9], [7, 2, 6, 13], [6, 38, 7], [15, 26, 17, 30], [17, 7, 30, 18], [7, 18, 23], [7], [16, 35, 38], [36, 38], [], [], [14, 2, 39, 6], [26], [], [], [19, 38, 7], [1, 13, 32, 15], [], [], [3, 38], [], [35, 27, 2, 37], [19, 10], [10, 18, 32, 7], [7, 18, 25], [11, 10, 35], [32], [], [21, 22, 35, 2], [21, 35, 2, 22], [], [35, 32, 1], [2, 19], [], [7, 23], [35, 3, 15, 23], [2], [28, 10, 29, 35]], [[35, 6, 23, 40], [35, 6, 22, 32], [14, 29, 10, 39], [10, 28, 24], [35, 2, 10, 31], [10, 18, 39, 31], [1, 29, 30, 36], [3, 39, 18, 31], [10, 13, 28, 38], [14, 15, 18, 40], [3, 36, 37, 10], [29, 35, 3, 5], [2, 14, 30, 40], [35, 28, 31, 40], [28, 27, 3, 18], [27, 16, 18, 38], [21, 36, 39, 31], [1, 6, 13], [35, 18, 24, 5], [28, 27, 12, 31], [28, 27, 18, 38], [35, 27, 2, 31], [], [], [15, 18, 35, 10], [6, 3, 10, 24], [10, 29, 39, 35], [16, 34, 31, 28], [35, 10, 24, 31], [33, 22, 30, 40], [10, 1, 34, 29], [15, 34, 33], [32, 28, 2, 24], [2, 35, 34, 27], [15, 10, 2], [35, 10, 28, 24], [35, 18, 10, 13], [35, 10, 18], [28, 35, 10, 23]], [[10, 24, 35], [10, 35, 5], [1, 26], [26], [30, 26], [30, 16], [], [2, 22], [26, 32], [], [], [], [], [], [10], [10], [], [19], [], [], [10, 19], [19, 10], [], [], [24, 26, 28, 32], [24, 28, 35], [10, 28, 23], [], [], [22, 10, 1], [10, 21, 22], [32], [27, 22], [], [], [], [35, 33], [35], [13, 23, 15]], [[10, 20, 37, 35], [10, 20, 26, 5], [15, 2, 29], [30, 24, 14, 5], [26, 4, 5, 16], [10, 35, 17, 4], [2, 5, 34, 10], [35, 16, 32, 18], [], [10, 37, 36, 5], [37, 36, 4], [4, 10, 34, 17], [35, 3, 22, 5], [29, 3, 28, 18], [20, 10, 28, 18], [28, 20, 10, 16], [35, 29, 21, 18], [1, 19, 26, 17], [35, 38, 19, 18], [1], [35, 20, 10, 6], [10, 5, 18, 32], [35, 18, 10, 39], [24, 26, 28, 32], [], [35, 38, 18, 16], [10, 30, 4], [24, 34, 28, 32], [24, 26, 28, 18], [35, 18, 34], [35, 22, 18, 39], [35, 28, 34, 4], [4, 28, 10, 34], [32, 1, 10], [35, 28], [6, 29], [18, 28, 32, 10], [24, 28, 35, 30], []], [[35, 6, 18, 31], [27, 26, 18, 35], [29, 14, 35, 18], [], [15, 14, 29], [2, 18, 40, 4], [15, 20, 29], [], [35, 29, 34, 28], [35, 14, 3], [10, 36, 14, 3], [35, 14], [15, 2, 17, 40], [14, 35, 34, 10], [3, 35, 10, 40], [3, 35, 31], [3, 17, 39], [], [34, 29, 16, 18], [3, 35, 31], [35], [7, 18, 25], [6, 3, 10, 24], [24, 28, 35], [35, 38, 18, 16], [], [18, 3, 28, 40], [13, 2, 28], [33, 30], [35, 33, 29, 31], [3, 35, 40, 39], [29, 1, 35, 27], [35, 29, 25, 10], [2, 32, 10, 25], [15, 3, 29], [3, 13, 27, 10], [3, 27, 29, 18], [8, 35], [13, 29, 3, 27]], [[3, 8, 10, 40], [3, 10, 8, 28], [15, 9, 14, 4], [15, 29, 28, 11], [17, 10, 14, 16], [32, 35, 40, 4], [3, 10, 14, 24], [2, 35, 24], [21, 35, 11, 28], [8, 28, 10, 3], [10, 24, 35, 19], [35, 1, 16, 11], [], [11, 28], [2, 35, 3, 25], [34, 27, 6, 40], [3, 35, 10], [11, 32, 13], [21, 11, 27, 19], [36, 23], [21, 11, 26, 31], [10, 11, 35], [10, 35, 29, 39], [10, 28], [10, 30, 4], [21, 28, 40, 3], [], [32, 3, 11, 23], [11, 32, 1], [27, 35, 2, 40], [35, 2, 40, 26], [], [27, 17, 40], [1, 11], [13, 35, 8, 24], [13, 35, 1], [27, 40, 28], [11, 13, 27], [1, 35, 29, 38]], [[32, 35, 26, 28], [28, 35, 25, 26], [28, 26, 5, 16], [32, 28, 3, 16], [26, 28, 32, 3], [26, 28, 32, 3], [32, 13, 6], [], [28, 13, 32, 24], [32, 2], [6, 28, 32], [6, 28, 32], [32, 35, 13], [28, 6, 32], [28, 6, 32], [10, 26, 24], [6, 19, 28, 24], [6, 1, 32], [3, 6, 32], [], [3, 6, 32], [26, 32, 27], [10, 16, 31, 28], [], [24, 34, 28, 32], [2, 6, 32], [5, 11, 1, 23], [], [], [28, 24, 22, 26], [3, 33, 39, 10], [6, 35, 25, 18], [1, 13, 17, 34], [1, 32, 13, 11], [13, 35, 2], [27, 35, 10, 34], [26, 24, 32, 28], [28, 2, 10, 34], [10, 34, 28, 32]], [[28, 32, 13, 18], [28, 35, 27, 9], [10, 28, 29, 37], [2, 32, 10], [28, 33, 29, 32], [2, 29, 18, 36], [32, 23, 2], [25, 10, 35], [10, 28, 32], [28, 19, 34, 36], [3, 35], [32, 30, 40], [30, 18], [3, 27], [3, 27, 40], [], [19, 26], [3, 32], [32, 2], [], [32, 2], [13, 32, 2], [35, 31, 10, 24], [], [32, 26, 28, 18], [32, 30], [11, 32, 1], [], [], [26, 28, 10, 36], [4, 17, 34, 26], [], [1, 32, 35, 23], [25, 10], [], [26, 2, 18], [], [26, 28, 18, 23], [10, 18, 32, 39]], [[22, 21, 27, 39], [2, 22, 13, 24], [17, 1, 39, 4], [1, 18], [22, 1, 33, 28], [27, 2, 39, 35], [22, 23, 37, 35], [34, 39, 19, 27], [21, 22, 35, 28], [13, 35, 39, 18], [22, 2, 37], [22, 1, 3, 35], [35, 24, 30, 18], [18, 35, 37, 1], [22, 15, 33, 28], [17, 1, 40, 33], [22, 33, 35, 2], [1, 19, 32, 13], [1, 24, 6, 27], [10, 2, 22, 37], [19, 22, 31, 2], [21, 22, 35, 2], [33, 22, 19, 40], [22, 10, 2], [35, 18, 34], [35, 33, 29, 31], [27, 24, 2, 40], [28, 33, 23, 26], [26, 28, 10, 18], [], [], [24, 35, 2], [2, 25, 28, 39], [35, 10, 2], [35, 11, 22, 31], [22, 19, 29, 40], [22, 19, 29, 40], [33, 3, 34], [22, 35, 13, 24]], [[19, 22, 15, 39], [35, 22, 1, 39], [17, 15, 16, 22], [], [17, 2, 18, 39], [22, 1, 40], [17, 2, 40], [30, 18, 35, 4], [35, 28, 3, 23], [35, 28, 1, 40], [2, 33, 27, 18], [35, 1], [35, 40, 27, 39], [15, 35, 22, 2], [15, 22, 33, 31], [21, 39, 16, 22], [22, 35, 2, 24], [19, 24, 39, 32], [2, 35, 6], [19, 22, 18], [2, 35, 18], [21, 35, 2, 22], [10, 1, 34], [10, 21, 29], [1, 22], [3, 24, 39, 1], [24, 2, 40, 39], [3, 33, 26], [4, 17, 34, 26], [], [], [], [], [], [], [19, 1, 31], [2, 21, 27, 1], [2], [22, 35, 18, 39]], [[28, 29, 15, 16], [1, 27, 36, 13], [1, 29, 13, 17], [15, 17, 27], [13, 1, 26, 12], [16, 40], [13, 29, 1, 40], [35], [35, 13, 8, 1], [35, 12], [35, 19, 1, 37], [1, 28, 13, 27], [11, 13, 1], [1, 3, 10, 32], [27, 1, 4], [35, 16], [27, 26, 18], [28, 24, 27, 1], [28, 26, 27, 1], [1, 4], [27, 1, 12, 24], [19, 35], [15, 34, 33], [32, 24, 18, 16], [35, 28, 34, 4], [35, 23, 1, 24], [], [1, 35, 12, 18], [], [24, 2], [], [], [2, 5, 13, 16], [35, 1, 11, 9], [2, 13, 15], [27, 26, 1], [6, 28, 11, 1], [8, 28, 1], [35, 1, 10, 28]], [[25, 2, 13, 15], [6, 13, 1, 25], [1, 17, 13, 12], [], [1, 17, 13, 16], [18, 16, 15, 39], [1, 16, 35, 15], [4, 18, 39, 31], [18, 13, 34], [28, 1335], [2, 32, 12], [15, 34, 29, 28], [32, 35, 30], [32, 40, 3, 28], [29, 3, 8, 25], [1, 16, 25], [26, 27, 13], [13, 17, 1, 24], [1, 13, 24], [], [35, 34, 2, 10], [2, 19, 13], [28, 32, 2, 24], [4, 10, 27, 22], [4, 28, 10, 34], [12, 35], [17, 27, 8, 40], [25, 13, 2, 34], [1, 32, 35, 23], [2, 25, 28, 39], [], [2, 5, 12], [], [12, 26, 1, 32], [15, 34, 1, 16], [32, 26, 12, 17], [], [1, 34, 12, 3], [15, 1, 28]], [[2, 2735, 11], [2, 27, 35, 11], [1, 28, 10, 25], [3, 18, 31], [15, 13, 32], [16, 25], [25, 2, 35, 11], [1], [34, 9], [1, 11, 10], [13], [1, 13, 2, 4], [2, 35], [11, 1, 2, 9], [11, 29, 28, 27], [1], [4, 10], [15, 1, 13], [15, 1, 28, 16], [], [15, 10, 32, 2], [15, 1, 32, 19], [2, 35, 34, 27], [], [32, 1, 10, 25], [2, 28, 10, 25], [11, 10, 1, 16], [10, 2, 13], [25, 10], [35, 10, 2, 16], [], [1, 35, 11, 10], [1, 12, 26, 15], [], [7, 1, 4, 16], [35, 1, 13, 11], [], [34, 35, 7, 13], [1, 32, 10]], [[1, 6, 15, 8], [19, 15, 29, 16], [35, 1, 29, 2], [1, 35, 16], [35, 30, 29, 7], [15, 16], [15, 35, 29], [], [35, 10, 14], [15, 17, 20], [35, 16], [15, 37, 1, 8], [35, 30, 14], [35, 3, 32, 6], [13, 1, 35], [2, 16], [27, 2, 3, 35], [6, 22, 26, 1], [19, 35, 29, 13], [], [19, 1, 29], [18, 15, 1], [15, 10, 2, 13], [], [35, 28], [3, 35, 15], [35, 13, 8, 24], [35, 5, 1, 10], [], [35, 11, 32, 31], [], [1, 13, 31], [15, 34, 1, 16], [1, 16, 7, 4], [], [15, 29, 37, 28], [1], [27, 34, 35], [35, 28, 6, 37]], [[26, 30, 34, 36], [2, 26, 35, 39], [1, 19, 26, 24], [26], [14, 1, 13, 16], [6, 36], [34, 26, 6], [1, 16], [34, 10, 28], [26, 16], [19, 1, 35], [29, 13, 28, 15], [2, 22, 17, 19], [2, 13, 28], [10, 4, 28, 15], [], [2, 17, 13], [24, 17, 13], [27, 2, 29, 28], [], [20, 19, 30, 34], [10, 35, 13, 2], [35, 10, 28, 29], [], [6, 29], [13, 3, 27, 10], [13, 35, 1], [2, 26, 10, 34], [26, 24, 32], [22, 19, 29, 40], [19, 1], [27, 26, 1, 13], [27, 9, 26, 24], [1, 13], [29, 15, 28, 37], [], [15, 10, 37, 28], [15, 1, 24], [12, 17, 28]], [[27, 26, 28, 13], [6, 13, 28, 1], [16, 17, 26, 24], [26], [2, 13, 18, 17], [2, 39, 30, 16], [29, 1, 4, 16], [2, 18, 26, 31], [3, 4, 16, 35], [30, 28, 40, 19], [35, 36, 37, 32], [27, 13, 1, 39], [11, 22, 39, 30], [27, 3, 15, 28], [19, 29, 39, 25], [25, 34, 6, 35], [3, 27, 35, 16], [2, 24, 26], [35, 38], [19, 35, 16], [18, 1, 16, 10], [35, 3, 15, 19], [1, 18, 10, 24], [35, 33, 27, 22], [18, 28, 32, 9], [3, 27, 29, 18], [27, 40, 28, 8], [26, 24, 32, 28], [], [22, 19, 29, 28], [2, 21], [5, 28, 11, 29], [2, 5], [12, 26], [1, 15], [15, 10, 37, 28], [], [34, 21], [35, 18]], [[28, 26, 18, 35], [28, 26, 35, 10], [14, 13, 17, 28], [23], [17, 14, 13], [], [35, 13, 16], [], [28, 10], [2, 35], [13, 35], [15, 32, 1, 13], [18, 1], [25, 13], [6, 9], [], [26, 2, 19], [8, 32, 19], [2, 32, 13], [], [28, 2, 27], [23, 28], [35, 10, 18, 5], [35, 33], [24, 28, 35, 30], [35, 13], [11, 27, 32], [28, 26, 10, 34], [28, 26, 18, 23], [2, 33], [2], [1, 26, 13], [1, 12, 34, 3], [1, 35, 13], [27, 4, 1, 35], [15, 24, 10], [34, 27, 25], [], [5, 12, 35, 26]], [[35, 26, 24, 37], [28, 27, 15, 3], [18, 4, 28, 38], [30, 7, 14, 26], [10, 26, 34, 31], [10, 35, 17, 7], [2, 6, 34, 10], [35, 37, 10, 2], [], [28, 15, 10, 36], [10, 37, 14], [14, 10, 34, 40], [35, 3, 22, 39], [29, 28, 10, 18], [35, 10, 2, 18], [20, 10, 16, 38], [35, 21, 28, 10], [26, 17, 19, 1], [35, 10, 38, 19], [1], [35, 20, 10], [28, 10, 29, 35], [28, 10, 35, 23], [13, 15, 23], [], [35, 38], [1, 35, 10, 38], [1, 10, 34, 28], [18, 10, 32, 1], [22, 35, 13, 24], [35, 22, 18, 39], [35, 28, 2, 24], [1, 28, 7, 10], [1, 32, 10, 25], [1, 35, 28, 37], [12, 17, 28, 24], [35, 18, 27, 2], [5, 12, 35, 26], []]
]

# -----------------------------------------------------------------------------
# App Pages
# -----------------------------------------------------------------------------

def contradiction_matrix_page(docs_dir, desired_model):
    """Renders the Contradiction Matrix tool page."""
    st.header("Contradiction Matrix Tool")
    st.markdown("""
    Use the dropdown menus to select the engineering parameter you want to improve and the one that gets worse. The tool will suggest relevant TRIZ principles.
    """)
    
    param_dict = {param: i for i, param in enumerate(TRIZ_PARAMETERS)}

    improving_feature = st.selectbox(
        "Feature to Improve:",
        options=TRIZ_PARAMETERS,
        index=8,
        key="improving_feature"
    )

    worsening_feature = st.selectbox(
        "Feature that Worsens:",
        options=TRIZ_PARAMETERS,
        index=0,
        key="worsening_feature"
    )

    st.header("Suggested TRIZ Principles")

    if improving_feature and worsening_feature:
        improving_idx = param_dict[improving_feature]
        worsening_idx = param_dict[worsening_feature]

        if improving_idx == worsening_idx:
            st.warning("The improving and worsening features cannot be the same.")
            return

        principle_numbers = CONTRADICTION_MATRIX[improving_idx][worsening_idx]

        st.info(f"**Improving:** `{improving_feature}`\n\n**Worsening:** `{worsening_feature}`")

        if not principle_numbers:
            st.success("No contradiction found for this pair according to the classical matrix.")
        else:
            for num in principle_numbers:
                if num in TRIZ_PRINCIPLES:
                    name, description = TRIZ_PRINCIPLES[num]
                    with st.expander(f"**Principle {num}: {name}**"):
                        st.write(description)
                        
                        # Ollama Integration
                        st.markdown("---")
                        if st.button(f"🤖 Get AI Explanation for Principle {num}", key=f"ollama_btn_{num}"):
                            try:
                                with st.spinner(f"Asking {desired_model} to explain..."):
                                    prompt = (
                                        f"You are an expert in the TRIZ methodology. Explain the TRIZ inventive principle '{name}' "
                                        f"(Principle {num}) in detail. The basic description is: '{description}'.\n\n"
                                        "Please elaborate on:\n1. The core concept.\n2. How it's applied.\n"
                                        "3. Provide at least two practical, real-world examples."
                                    )
                                    ai_text = chat_text(prompt, model=desired_model)
                                    st.markdown("### AI-Generated Explanation")
                                    st.info(ai_text)
                            except Exception as e:
                                st.error(f"Could not connect to Ollama. Ensure it's running and the model '{desired_model}' is available. Error: {e}")


                        # Local File Integration
                        principle_dir = os.path.join(docs_dir, f"principle_{num}")
                        if os.path.isdir(principle_dir):
                            st.markdown("---")
                            st.subheader("Supporting Documents & Images")
                            content_path_md = os.path.join(principle_dir, "content.md")
                            content_path_txt = os.path.join(principle_dir, "content.txt")
                            if os.path.exists(content_path_md):
                                with open(content_path_md, "r", encoding="utf-8") as f:
                                    st.markdown(f.read(), unsafe_allow_html=True)
                            elif os.path.exists(content_path_txt):
                                with open(content_path_txt, "r", encoding="utf-8") as f:
                                    st.text(f.read())
                            
                            image_files = glob.glob(os.path.join(principle_dir, "*.png")) + glob.glob(os.path.join(principle_dir, "*.jpg")) + glob.glob(os.path.join(principle_dir, "*.jpeg"))
                            if image_files:
                                for image_path in sorted(image_files):
                                    st.image(image_path)

def ai_problem_solver_page(desired_model):
    """Renders the AI Problem Solver page."""
    st.header("AI Problem Solver")
    st.markdown("""
    Describe your problem and the contradiction you are facing. The AI will analyze your input and suggest relevant TRIZ principles and potential solution concepts.
    """)

    problem_statement = st.text_area("Enter your problem statement:", height=150, placeholder="e.g., 'I need to design a coffee cup that keeps coffee hot for a long time but is not hot to the touch.'")
    contradiction = st.text_area("Describe the contradiction:", height=100, placeholder="e.g., 'The cup needs to have high thermal insulation to keep the coffee hot, but this usually makes the outer surface hot and unsafe to hold.'")

    if st.button("Analyze with AI", key="analyze_problem"):
        if not problem_statement or not contradiction:
            st.warning("Please fill in both the problem statement and the contradiction.")
        else:
            try:
                with st.spinner(f"Asking {desired_model} to analyze your problem..."):
                    prompt = (
                        "You are a world-class TRIZ expert. Analyze the following engineering problem and its contradiction. "
                        "Your task is to identify the most relevant TRIZ inventive principles that could solve this problem. "
                        "For each principle you identify, you must:\n"
                        "1. Name the principle and its number.\n"
                        "2. Briefly explain why it is applicable to this specific problem.\n"
                        "3. Provide a concrete example of a solution concept based on that principle for the given problem.\n\n"
                        f"**Problem Statement:**\n{problem_statement}\n\n"
                        f"**Contradiction:**\n{contradiction}\n\n"
                        "Provide your analysis clearly and concisely, focusing only on TRIZ-based solutions."
                    )
                    ai_text = chat_text(prompt, model=desired_model)
                    st.markdown("### AI-Powered TRIZ Analysis")
                    st.success(ai_text)
            except Exception as e:
                st.error(f"Could not connect to Ollama. Please ensure Ollama is running and the model '{desired_model}' is available. Error: {e}")


def patent_search_page():
    """Renders the AI-Powered Patent Search page."""
    st.header("AI-Powered Patent Search")
    st.markdown("""
    Upload your patent database (CSV or Excel file) and describe your problem. The AI will find the most conceptually relevant patents from your file.
    """)

    uploaded_file = st.file_uploader("Upload your patent CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_patent_data(uploaded_file)
        
        if df is not None:
            st.success(f"Successfully loaded {len(df)} patents.")
            
            problem_statement = st.text_area("Enter a detailed problem statement:", height=150, key="patent_problem")
            
            relevance_threshold = st.slider("Set Relevance Threshold", min_value=0.5, max_value=1.0, value=0.75, step=0.05)

            if st.button("Search for Relevant Patents"):
                if not problem_statement:
                    st.warning("Please enter a problem statement.")
                else:
                    with st.spinner("Analyzing patents... This may take a moment."):
                        model = load_sentence_model()
                        patent_embeddings = generate_patent_embeddings(df, model)
                        
                        if patent_embeddings is None:
                            st.error("Could not find an 'abstract' column in the uploaded file. Please ensure the column exists and is named correctly (case-insensitive).")
                            return

                        problem_embedding = model.encode(problem_statement, convert_to_tensor=True)
                        
                        # Compute cosine similarity
                        cosine_scores = util.cos_sim(problem_embedding, patent_embeddings)
                        
                        df['relevance'] = cosine_scores[0].cpu().numpy()
                        
                        # Filter based on threshold
                        
                        relevant_patents = df[df['relevance'] >= relevance_threshold].sort_values(by='relevance', ascending=False)

                        st.markdown("---")
                        st.header("Search Results")

                        import pandas as pd
                        import json
                        def analyze_patent_with_llama3(title, abstract, claims, model):
                            prompt = f"""
                        Analyze the following patent:
                        
                        Title: {title}
                        Abstract: {abstract}
                        Claims: {claims}
                        
                        Answer these questions as a JSON object. Keep every answer as short, direct, and specific as possible:
                        {{
                          "Topic": "<1-2 word topic>",
                          "Domain": "<specific innovation domain>",
                          "Ecosystem": "<single word for main ecosystem>",
                          "Categorization": "<category> / <subcategory> / <primary issue> / <solution>",
                          "NewTech": "<short phrase or 'none'>",
                          "ProblemAlt": "<problem summarized>; <1 creative alternate solution>",
                          "Summary": "<single CLEAR summary sentence>"
                        }}
                        Your entire response must be valid JSON.
                        """.strip()
                            raw_content = chat_text(prompt, model=model)
                            json_start = raw_content.find("{")
                            json_end = raw_content.rfind("}")
                            try:
                                json_str = raw_content[json_start:json_end+1]
                                import json
                                answers = json.loads(json_str)
                            except Exception:
                                answers = {
                                    "Topic": "", "Domain": "", "Ecosystem": "",
                                    "Categorization": "", "NewTech": "",
                                    "ProblemAlt": "", "Summary": ""
                                }
                            return answers



    # ---------- Streamlit UI ----------
    #st.set_page_config(page_title="Problem Canvas Generator", layout="wide")

def problem_statment_canvas_page(desired_model):
    def call_llama(prompt: str) -> str:
        client = get_ollama_client()
        resp = client.chat(
            model=desired_model,
            messages=[{"role": "user", "content": prompt.strip()}],
            stream=False
        )
        return resp["message"]["content"].strip()

    def build_prompt(ctx, prob, cust, emo, quant, alt, alt_sc):
        return f"""
You are an expert problem-statement writer.

TASK: Combine EVERY fact below into ONE clear paragraph, in this order:
1. Context (trigger) → 2. Problem (lose / pain) → 3. Impact on life → 4. Motivation/emotion → 
5. Quantifiable impact (numbers) → 6. Current alternative (proof) → 7. Its shortcoming (opportunity / competitive advantage).

Do NOT label the parts. Do NOT add headings. Use plain sentences separated by commas or semicolons as needed. 
After the paragraph, add one blank line and then output a short bullet list recap of the key facts.

Facts:
Context: {ctx}
Problem: {prob}
Customers: {cust}
Emotional impact: {emo}
Quantifiable impact: {quant}
Current alternative: {alt}
Alternative shortcomings: {alt_sc}
"""
    # UI continues unchanged...

    
    

    
    st.markdown(
        "<h2 style='margin-bottom:1rem'>Problem Statement Canvas</h2>",
        unsafe_allow_html=True,
    )
    
    # --- Row 1 (Context | Problem | Alternatives) ---
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**CONTEXT**  \n<small>When does the problem occur?</small>", unsafe_allow_html=True)
            ctx = st.text_area(" ", key="ctx", height=90)
    
        with col2:
            st.markdown("**PROBLEM**  \n<small>What is the root cause?</small>", unsafe_allow_html=True)
            prob = st.text_area("  ", key="prob", height=90)
    
        with col3:
            st.markdown("**ALTERNATIVES**  \n<small>What do customers do now?</small>", unsafe_allow_html=True)
            alt = st.text_area("   ", key="alt", height=90)
    
    # --- Row 2 (Customers | Emotional | Alt-Shortcomings) ---
    with st.container():
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("**CUSTOMERS**  \n<small>Who has the problem most often?</small>", unsafe_allow_html=True)
            cust = st.text_area("    ", key="cust", height=90)
    
        with col5:
            st.markdown("**EMOTIONAL IMPACT**  \n<small>How do they feel?</small>", unsafe_allow_html=True)
            emo = st.text_area("     ", key="emo", height=90)
    
        with col6:
            st.markdown("**ALTERNATIVE SHORTCOMINGS**  \n<small>Disadvantages?</small>", unsafe_allow_html=True)
            alt_sc = st.text_area("      ", key="alt_sc", height=90)
    
    # --- Row 3 (Quantifiable Impact) ---
    with st.container():
        st.markdown("**QUANTIFIABLE IMPACT**  \n<small>Measurable impact (include units)</small>", unsafe_allow_html=True)
        quant = st.text_area("       ", key="quant", height=70)
    
    st.divider()
    
    # ---------- Generate button ----------
    if st.button("Generate statement 📝", use_container_width=True):
        if not any([ctx, prob, cust, emo, quant, alt, alt_sc]):
            st.warning("Please fill at least one field.")
        else:
            with st.spinner("Calling Llama 3-8B …"):
                prompt = build_prompt(ctx, prob, cust, emo, quant, alt, alt_sc)
                result = call_llama(prompt)
    
            st.success("Done!")
            st.markdown(
            f"""
<div style="
    width:100%;
    overflow-wrap:break-word;
    white-space:pre-wrap;
    font-family:monospace;
    border:1px solid #ddd;
    border-radius:6px;
    background:#f7f7f7;
    padding:12px;
    max-height:350px;          /* grow down, then vertical-scroll */
    overflow-y:auto;
">
{result}
</div>
""",
            unsafe_allow_html=True,
        )
            #st.code(result, language="text")
    
    # ---------- Sidebar ----------
     # Instructions section at the very bottom
    with st.expander("📋 How to Use This Tool"):
        st.markdown("""
            1. Fill the six boxes following the canvas.
            2. Click **Generate statement**.""")
            #3. Copy the single-paragraph result (plus bullet recap).""")


def root_conflict_analysis_page():

    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, Circle
    import numpy as np
    from typing import Dict, List, Tuple
    import io

        # Set page config
    #st.set_page_config(page_title="Manual RCA+ Tool", layout="wide")

    def initialize_session_state():
        """Initialize session state with default data"""
        if 'causes_data' not in st.session_state:
            st.session_state.causes_data = pd.DataFrame({
                'ID': ['MAIN', 'C1', 'C2'],
                'Cause/Effect Name': [
                    'Main Negative Effect', 
                    'Example Cause 1', 
                    'Example Cause 2'
                ],
                'Type': ['N', 'N+P', 'N'],
                'Positive Effect': ['-', 'Saves cost', '-'],
                'Negative Effect': [
                    'Primary problem statement', 
                    'Creates delay', 
                    'Reduces quality'
                ],
                'Caused By (IDs)': ['-', 'MAIN', 'C1'],
                'Causes (IDs)': ['C1,C2', '-', '-'],
                'Relationship': ['OR', 'AND', 'AND'],
                'Factual/Assumptive': ['Factual', 'Factual', 'Assumptive'],
                'Scientific Reasoning': [
                    'Main problem to solve',
                    'Cost-benefit trade-off creates contradiction',
                    'Secondary effect from C1'
                ]
            })

    def validate_csv_format(df: pd.DataFrame) -> tuple:
        """Validate uploaded CSV format and return (is_valid, error_message, cleaned_df)"""
        required_columns = [
            'ID', 'Cause/Effect Name', 'Type', 'Positive Effect', 'Negative Effect',
            'Caused By (IDs)', 'Causes (IDs)', 'Relationship', 'Factual/Assumptive', 
            'Scientific Reasoning'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", None
        
        # Clean the dataframe
        cleaned_df = df[required_columns].copy()
        
        # Validate Type column values
        valid_types = ['N', 'N+P', 'NC', 'P']
        invalid_types = cleaned_df[~cleaned_df['Type'].isin(valid_types)]['Type'].unique()
        if len(invalid_types) > 0:
            return False, f"Invalid Type values found: {', '.join(invalid_types)}. Must be one of: {', '.join(valid_types)}", None
        
        # Validate Relationship column values
        valid_relationships = ['AND', 'OR']
        invalid_relationships = cleaned_df[~cleaned_df['Relationship'].isin(valid_relationships)]['Relationship'].unique()
        if len(invalid_relationships) > 0:
            return False, f"Invalid Relationship values found: {', '.join(invalid_relationships)}. Must be 'AND' or 'OR'", None
        
        # Validate Factual/Assumptive column values
        valid_factual = ['Factual', 'Assumptive']
        invalid_factual = cleaned_df[~cleaned_df['Factual/Assumptive'].isin(valid_factual)]['Factual/Assumptive'].unique()
        if len(invalid_factual) > 0:
            return False, f"Invalid Factual/Assumptive values found: {', '.join(invalid_factual)}. Must be 'Factual' or 'Assumptive'", None
        
        # Fill NaN values with appropriate defaults
        cleaned_df['Positive Effect'] = cleaned_df['Positive Effect'].fillna('-')
        cleaned_df['Negative Effect'] = cleaned_df['Negative Effect'].fillna('-')
        cleaned_df['Caused By (IDs)'] = cleaned_df['Caused By (IDs)'].fillna('-')
        cleaned_df['Causes (IDs)'] = cleaned_df['Causes (IDs)'].fillna('-')
        cleaned_df['Scientific Reasoning'] = cleaned_df['Scientific Reasoning'].fillna('')
        
        # Check for duplicate IDs
        duplicate_ids = cleaned_df[cleaned_df['ID'].duplicated()]['ID'].unique()
        if len(duplicate_ids) > 0:
            return False, f"Duplicate IDs found: {', '.join(duplicate_ids)}", None
        
        return True, "CSV format is valid", cleaned_df
    
    def load_csv_data(uploaded_file) -> tuple:
        """Load and validate CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate format
            is_valid, message, cleaned_df = validate_csv_format(df)
            
            if is_valid:
                return True, f"✅ CSV loaded successfully! Found {len(cleaned_df)} causes.", cleaned_df
            else:
                return False, f"❌ CSV validation failed: {message}", None
                
        except Exception as e:
            return False, f"❌ Error reading CSV file: {str(e)}", None
    
    def create_rca_diagram(df: pd.DataFrame) -> plt.Figure:
        """Create RCA+ flow diagram from causes dataframe"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Color mapping for different cause types
        color_map = {
            'N': '#ffcccc',      # Light red for negative
            'N+P': '#ffd700',    # Gold for contradictions  
            'NC': '#d3d3d3',     # Gray for non-changeable
            'P': '#ccffcc',      # Light green for positive
            'MAIN': '#ff9999'    # Darker red for main effect
        }
        
        # Symbol mapping
        symbol_map = {
            'N': '(−)',
            'N+P': '(±)',
            'NC': '(−−)',
            'P': '(+)',
            'MAIN': ''
        }
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for _, row in df.iterrows():
            node_type = 'MAIN' if row['ID'] == 'MAIN' else row['Type']
            G.add_node(
                row['ID'],
                label=row['Cause/Effect Name'],
                type=node_type,
                pos_effect=row['Positive Effect'],
                neg_effect=row['Negative Effect'],
                relationship=row['Relationship'],
                factual=row['Factual/Assumptive']
            )
        
        # Add edges based on "Caused By" relationships
        for _, row in df.iterrows():
            if pd.notna(row['Caused By (IDs)']) and row['Caused By (IDs)'] != '-':
                parent_ids = [pid.strip() for pid in str(row['Caused By (IDs)']).split(',')]
                for parent_id in parent_ids:
                    if parent_id in G.nodes():
                        G.add_edge(parent_id, row['ID'])
        
        # Create hierarchical layout
        pos = create_hierarchical_layout(G, df)
        
        # Clear the axes
        ax.clear()
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 8.5)
        ax.axis('off')
        
        # Draw nodes
        for node_id, node_data in G.nodes(data=True):
            if node_id in pos:
                x, y = pos[node_id]
                node_type = node_data['type']
                color = color_map.get(node_type, '#ffffff')
                
                # Get row data for this node
                node_row = df[df['ID'] == node_id].iloc[0]
                
                # Create fancy box for node
                width = 2.0 if node_type == 'MAIN' else 1.8
                height = 1.2 if node_type == 'MAIN' else 1.0
                
                # Different line style for assumptive causes
                line_style = '--' if node_data['factual'] == 'Assumptive' else '-'
                
                bbox = FancyBboxPatch(
                    (x - width/2, y - height/2), width, height,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2 if node_type == 'MAIN' else 1.5,
                    linestyle=line_style
                )
                ax.add_patch(bbox)
                
                # Add node text
                label = node_data['label']
                if len(label) > 25:
                    label = label[:22] + "..."
                
                symbol = symbol_map.get(node_type, '')
                text_size = 10 if node_type == 'MAIN' else 8
                
                ax.text(x, y, f"{symbol}\n{label}", 
                       ha='center', va='center', fontsize=text_size, 
                       weight='bold' if node_type == 'MAIN' else 'normal',
                       wrap=True)
                
                # Add positive effect if exists and is contradiction
                if node_type == 'N+P' and node_data['pos_effect'] != '-':
                    ax.text(x + width/2 + 0.1, y + 0.3, f"(+) {node_data['pos_effect'][:20]}", 
                           ha='left', va='center', fontsize=7, style='italic',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
                
                # Add relationship indicator for AND relationships
                if node_row['Relationship'] == 'AND':
                    circle = Circle((x, y - height/2 - 0.3), 0.15, 
                                  facecolor='white', edgecolor='black', linewidth=2)
                    ax.add_patch(circle)
                    ax.text(x, y - height/2 - 0.3, 'AND', ha='center', va='center', fontsize=6, weight='bold')
        
        # Draw edges with arrows
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x1, y1 = pos[edge[0]]
                x2, y2 = pos[edge[1]]
                
                # Calculate arrow position to node edge
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Adjust start and end points to box edges
                    start_x = x1 + (dx/length) * 0.9
                    start_y = y1 + (dy/length) * 0.6
                    end_x = x2 - (dx/length) * 0.9
                    end_y = y2 - (dy/length) * 0.6
                    
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Add legend
        legend_elements = [
            patches.Patch(color='#ff9999', label='Main Negative Effect'),
            patches.Patch(color='#ffcccc', label='Negative Cause (−)'),
            patches.Patch(color='#ffd700', label='Contradiction (±)'),
            patches.Patch(color='#d3d3d3', label='Non-changeable (−−)'),
            patches.Patch(color='#ccffcc', label='Positive Effect (+)'),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Assumptive')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add title
        plt.title("Root Conflict Analysis (RCA+) Flow Diagram", fontsize=16, weight='bold', pad=20)
        
        return fig

    def create_hierarchical_layout(G: nx.DiGraph, df: pd.DataFrame) -> Dict:
        """Create hierarchical layout for the graph"""
        pos = {}
        
        # Find root nodes (nodes with no incoming edges)
        root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not root_nodes:
            root_nodes = ['MAIN'] if 'MAIN' in G.nodes() else list(G.nodes())[:1]
        
        # Assign levels using BFS
        levels = {}
        queue = [(node, 0) for node in root_nodes]
        visited = set()
        
        while queue:
            node, level = queue.pop(0)
            if node not in visited:
                visited.add(node)
                levels[node] = level
                
                # Add successors to queue
                for successor in G.successors(node):
                    if successor not in visited:
                        queue.append((successor, level + 1))
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Position nodes
        max_level = max(level_groups.keys()) if level_groups else 0
        for level, nodes in level_groups.items():
            y_pos = 7 - (level * 2)  # Top to bottom
            
            if len(nodes) == 1:
                pos[nodes[0]] = (5, y_pos)  # Center single nodes
            else:
                # Distribute multiple nodes horizontally
                x_positions = np.linspace(1, 9, len(nodes))
                for i, node in enumerate(nodes):
                    pos[node] = (x_positions[i], y_pos)
        
        return pos
    
    def export_table_to_csv(df: pd.DataFrame) -> str:
        """Export dataframe to CSV string"""
        return df.to_csv(index=False)
    
    def main():
        # Initialize session state
        initialize_session_state()
        
        st.title("🔬 Manual Root Conflict Analysis (RCA+) Tool")
        st.markdown("*Create and visualize RCA+ diagrams by editing the cause table below*")
        
        # ========== SECTION 1: CAUSES TABLE ==========
        st.header("📝 RCA+ Causes Table")
        st.markdown("**Edit the table below to build your RCA+ analysis:**")
        
        # CSV Upload Section
        with st.expander("📂 Upload RCA+ CSV File", expanded=False):
            st.markdown("**Upload a previously saved RCA+ analysis CSV file:**")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with RCA+ cause table data"
            )
            
            if uploaded_file is not None:
                col_upload1, col_upload2 = st.columns([2, 1])
                
                with col_upload1:
                    # Load and validate the CSV
                    success, message, loaded_df = load_csv_data(uploaded_file)
                    
                    if success:
                        st.success(message)
                        
                        # Preview the loaded data
                        st.subheader("📋 Preview of uploaded data:")
                        st.dataframe(loaded_df.head(), use_container_width=True)
                        
                    else:
                        st.error(message)
                        st.write("**Please ensure your CSV has these columns:**")
                        st.code("""
    ID, Cause/Effect Name, Type, Positive Effect, Negative Effect, 
    Caused By (IDs), Causes (IDs), Relationship, Factual/Assumptive, 
    Scientific Reasoning
                        """)
                
                with col_upload2:
                    if 'loaded_df' in locals() and loaded_df is not None:
                        if st.button("✅ Load This Data", type="primary"):
                            st.session_state.causes_data = loaded_df
                            st.success("🎉 Data loaded successfully into the table!")
                            st.rerun()
                    
                    if st.button("📥 Download Sample CSV"):
                        sample_csv = export_table_to_csv(st.session_state.causes_data)
                        st.download_button(
                            label="💾 Sample CSV Template",
                            data=sample_csv,
                            file_name="rca_sample_template.csv",
                            mime="text/csv"
                        )
        
        # Display editable dataframe
        edited_df = st.data_editor(
            st.session_state.causes_data,
            column_config={
                "ID": st.column_config.TextColumn(
                    "ID",
                    help="Unique identifier for each cause",
                    width="small"
                ),
                "Cause/Effect Name": st.column_config.TextColumn(
                    "Cause/Effect Name",
                    help="Description of the cause or effect",
                    width="large"
                ),
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    help="N=Negative, N+P=Contradiction, NC=Non-changeable, P=Positive",
                    options=["N", "N+P", "NC", "P"],
                    width="small"
                ),
                "Positive Effect": st.column_config.TextColumn(
                    "Positive Effect",
                    help="Beneficial effect (use '-' if none)",
                    width="medium"
                ),
                "Negative Effect": st.column_config.TextColumn(
                    "Negative Effect", 
                    help="Harmful effect (use '-' if none)",
                    width="medium"
                ),
                "Caused By (IDs)": st.column_config.TextColumn(
                    "Caused By (IDs)",
                    help="Parent cause IDs (comma-separated)",
                    width="small"
                ),
                "Causes (IDs)": st.column_config.TextColumn(
                    "Causes (IDs)",
                    help="Child effect IDs (comma-separated)", 
                    width="small"
                ),
                "Relationship": st.column_config.SelectboxColumn(
                    "Relationship",
                    help="AND=All causes needed, OR=Any cause sufficient",
                    options=["AND", "OR"],
                    width="small"
                ),
                "Factual/Assumptive": st.column_config.SelectboxColumn(
                    "Factual/Assumptive",
                    help="Is this cause verified (Factual) or hypothetical (Assumptive)?",
                    options=["Factual", "Assumptive"],
                    width="small"
                ),
                "Scientific Reasoning": st.column_config.TextColumn(
                    "Scientific Reasoning",
                    help="Explanation using scientific principles",
                    width="large"
                )
            },
            num_rows="dynamic",
            use_container_width=True,
            key="causes_editor"
        )
        
        # Update session state
        st.session_state.causes_data = edited_df
        
        # Action buttons
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            if st.button("➕ Add Sample Row"):
                new_row = pd.DataFrame({
                    'ID': [f'C{len(edited_df)}'],
                    'Cause/Effect Name': ['New Cause'],
                    'Type': ['N'],
                    'Positive Effect': ['-'],
                    'Negative Effect': ['Describe negative effect'],
                    'Caused By (IDs)': ['-'],
                    'Causes (IDs)': ['-'],
                    'Relationship': ['AND'],
                    'Factual/Assumptive': ['Factual'],
                    'Scientific Reasoning': ['Add reasoning here']
                })
                st.session_state.causes_data = pd.concat([edited_df, new_row], ignore_index=True)
                st.rerun()
        
        with col_b:
            if st.button("🔄 Refresh Diagram"):
                st.rerun()
        
        with col_c:
            if st.button("🗑️ Clear All Data"):
                if st.session_state.get('confirm_clear', False):
                    # Reset to initial state
                    st.session_state.causes_data = pd.DataFrame({
                        'ID': ['MAIN'],
                        'Cause/Effect Name': ['Main Negative Effect'],
                        'Type': ['N'],
                        'Positive Effect': ['-'],
                        'Negative Effect': ['Primary problem statement'],
                        'Caused By (IDs)': ['-'],
                        'Causes (IDs)': ['-'],
                        'Relationship': ['OR'],
                        'Factual/Assumptive': ['Factual'],
                        'Scientific Reasoning': ['Main problem to solve']
                    })
                    st.session_state.confirm_clear = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm clearing all data")
        
        with col_d:
            csv_data = export_table_to_csv(edited_df)
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name="rca_analysis.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        # ========== SECTION 2: FLOW DIAGRAM ==========
        st.header("🌊 RCA+ Flow Diagram")
        
        try:
            fig = create_rca_diagram(edited_df)
            st.pyplot(fig, use_container_width=True)
            
            # Save diagram button
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="💾 Download Diagram (PNG)",
                data=buf,
                file_name="rca_diagram.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error creating diagram: {str(e)}")
            st.write("Please check your table data for errors.")
        
        st.divider()
        
        # ========== SECTION 3: ANALYSIS SUMMARY, KEY CONTRADICTIONS, VALIDATION ==========
        
        # Create three columns for the bottom analysis section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.header("📊 Analysis Summary")
            
            # Summary statistics
            total_causes = len(edited_df)
            contradictions = len(edited_df[edited_df['Type'] == 'N+P'])
            negative_causes = len(edited_df[edited_df['Type'] == 'N'])
            non_changeable = len(edited_df[edited_df['Type'] == 'NC'])
            positive_causes = len(edited_df[edited_df['Type'] == 'P'])
            factual_count = len(edited_df[edited_df['Factual/Assumptive'] == 'Factual'])
            assumptive_count = len(edited_df[edited_df['Factual/Assumptive'] == 'Assumptive'])
            
            # Display metrics in a nice format
            st.metric("Total Causes", total_causes)
            
            # Create two sub-columns for better layout
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.metric("Contradictions", contradictions)
                st.metric("Negative", negative_causes)
                st.metric("Positive", positive_causes)
            with sub_col2:
                st.metric("Non-changeable", non_changeable)
                st.metric("Factual", factual_count)
                st.metric("Assumptive", assumptive_count)
            
            # Analysis insights
            st.subheader("📈 Insights")
            if contradictions > 0:
                st.success(f"Found {contradictions} contradiction(s) - key focus areas!")
            if assumptive_count > 0:
                st.info(f"{assumptive_count} cause(s) need verification")
            if non_changeable > 0:
                st.warning(f"{non_changeable} cause(s) are beyond control")
        
        with col2:
            st.header("🎯 Key Contradictions")
            
            if contradictions > 0:
                contradiction_df = edited_df[edited_df['Type'] == 'N+P']
                
                for i, (_, row) in enumerate(contradiction_df.iterrows(), 1):
                    st.subheader(f"Contradiction #{i}")
                    
                    # Create a nice box for each contradiction
                    with st.container():
                        st.write(f"**{row['ID']}:** {row['Cause/Effect Name']}")
                        
                        # Show positive and negative effects with icons
                        col_pos, col_neg = st.columns(2)
                        with col_pos:
                            st.write(f"✅ **Positive:** {row['Positive Effect']}")
                        with col_neg:
                            st.write(f"❌ **Negative:** {row['Negative Effect']}")
                        
                        if row['Scientific Reasoning'] != '' and pd.notna(row['Scientific Reasoning']):
                            st.write(f"🔬 **Reasoning:** {row['Scientific Reasoning']}")
                        
                        st.write("---")
            else:
                st.info("No contradictions found. Add causes with type 'N+P' to identify contradictions.")
                st.write("**Contradictions are key in RCA+ analysis as they represent:**")
                st.write("- Causes that create both positive and negative effects")
                st.write("- Areas where innovative solutions are needed")
                st.write("- Trade-offs that require careful balance")
        
        with col3:
            st.header("⚠️ Validation")
            
            warnings = []
            suggestions = []
            
            # Check for missing main effect
            if 'MAIN' not in edited_df['ID'].values:
                warnings.append("❌ No MAIN negative effect defined")
            else:
                suggestions.append("✅ MAIN effect properly defined")
            
            # Check for orphaned causes
            all_ids = set(edited_df['ID'].values)
            orphaned_refs = []
            
            for _, row in edited_df.iterrows():
                if pd.notna(row['Caused By (IDs)']) and row['Caused By (IDs)'] != '-':
                    parent_ids = [pid.strip() for pid in str(row['Caused By (IDs)']).split(',')]
                    for pid in parent_ids:
                        if pid not in all_ids and pid not in orphaned_refs:
                            orphaned_refs.append(pid)
            
            if orphaned_refs:
                for ref in orphaned_refs:
                    warnings.append(f"❌ Referenced ID '{ref}' not found in table")
            else:
                suggestions.append("✅ All ID references are valid")
            
            # Check for isolated nodes
            isolated_nodes = []
            for _, row in edited_df.iterrows():
                if (row['ID'] != 'MAIN' and 
                    (pd.isna(row['Caused By (IDs)']) or row['Caused By (IDs)'] == '-') and
                    (pd.isna(row['Causes (IDs)']) or row['Causes (IDs)'] == '-')):
                    isolated_nodes.append(row['ID'])
            
            if isolated_nodes:
                warnings.append(f"⚠️ Isolated nodes found: {', '.join(isolated_nodes)}")
            
            # Check for contradictions without positive effects
            for _, row in edited_df.iterrows():
                if row['Type'] == 'N+P' and (pd.isna(row['Positive Effect']) or row['Positive Effect'] == '-'):
                    warnings.append(f"❌ {row['ID']}: Contradiction missing positive effect")
            
            # Display results
            if warnings:
                st.subheader("⚠️ Issues Found:")
                for warning in warnings:
                    st.write(warning)
            
            if suggestions:
                st.subheader("✅ Validation Passed:")
                for suggestion in suggestions:
                    st.write(suggestion)
            
            if not warnings:
                st.success("🎉 All validations passed!")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if contradictions == 0:
                st.write("• Look for causes that have both positive and negative effects")
            if assumptive_count > factual_count:
                st.write("• Verify assumptive causes with data/testing")
            if non_changeable > contradictions:
                st.write("• Focus on changeable causes for solution development")
        
        # Instructions section at the very bottom
        with st.expander("📋 How to Use This Tool"):
            st.markdown("""
            ### Step-by-Step Instructions:
            
            #### 🆕 Upload/Download Data:
            1. **Upload CSV**: Use the "Upload RCA+ CSV File" section to load previously saved analyses
            2. **Download CSV**: Save your current work using the "Download CSV" button
            3. **Sample Template**: Download a sample CSV to see the correct format
            
            #### ✏️ Edit Analysis:
            1. **Start with Main Effect**: Keep or modify the 'MAIN' row with your primary problem
            2. **Add Causes**: Click "Add Sample Row" or edit existing rows to add causes
            3. **Set Relationships**: 
               - **ID**: Unique identifiers (C1, C2, etc.)
               - **Type**: N (negative), N+P (contradiction), NC (non-changeable), P (positive)
               - **Caused By**: Enter parent cause IDs (comma-separated)
               - **Relationship**: AND (all causes needed) or OR (any cause sufficient)
            4. **Define Effects**:
               - **Positive Effect**: Benefits from this cause (for contradictions)
               - **Negative Effect**: Problems caused
            5. **Add Reasoning**: Scientific explanation for each cause
            
            ### CSV File Format:
            Your CSV must have these exact column headers:
            ```
            ID, Cause/Effect Name, Type, Positive Effect, Negative Effect, 
            Caused By (IDs), Causes (IDs), Relationship, Factual/Assumptive, 
            Scientific Reasoning
            ```
            
            ### RCA+ Types:
            - **N (Negative)**: Pure harmful effects to eliminate
            - **N+P (Contradiction)**: Same cause creates both positive and negative effects ⭐
            - **NC (Non-changeable)**: Harmful causes beyond your control
            - **P (Positive)**: Beneficial effects
            
            ### Visual Indicators:
            - **Solid lines**: Factual causes
            - **Dashed lines**: Assumptive causes (need verification)
            - **Colors**: Different types have different colors in the diagram
            
            **💡 Pro Tip:** Focus on contradictions (N+P) as they represent the most innovative solution opportunities!
            """)
    
    if __name__ == "__main__":
        main()

# -----------------------------------------------------------------------------
# Main App Logic
# -----------------------------------------------------------------------------

def main():
    # Sidebar brand
    # Sidebar brand
    #st.sidebar.title("InnoNexus")
    
    # Ollama config (give every widget a unique key)


    

    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Choose a tool",
        ["Problem Statement Canvas", "Patent Search", "RCA+", "Contradiction Matrix", "AI Problem Solver"]
    )

    # Robust path for docs_dir if needed by a page
    import os
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    docs_dir = os.path.join(script_dir, "triz_docs")

    # Route to pages (these functions must NOT call st.set_page_config)
    if page == "Problem Statement Canvas":
        problem_statment_canvas_page(desired_model)
    elif page == "Patent Search":
        patent_search_page()
    elif page == "RCA+":
        root_conflict_analysis_page()
    elif page == "Contradiction Matrix":
        contradiction_matrix_page(docs_dir, desired_model)
    elif page == "AI Problem Solver":
        ai_problem_solver_page(desired_model)

# Ensure main() actually runs
if __name__ == "__main__":
    main()

