import os
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    # Show debugging info
    st.write("**Debug Information:**")
    st.write(f"Current working directory: `{os.getcwd()}`")
    st.write(f"Script location: `{__file__}`")
    
    # List all files in current directory
    st.write("**Files in current directory:**")
    files = os.listdir('.')
    st.write(files)
    
    # Try to find pkl files
    pkl_files = [f for f in files if f.endswith('.pkl')]
    st.write(f"**Found .pkl files:** {pkl_files}")
    
    try:
        model = joblib.load('breastcancermodel.pkl')
        scaler = joblib.load('scaler.pkl')
        st.success("✅ Model loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None

model, scaler = load_model()
