from src.webapp.app import MedicalQAApp
import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    app = MedicalQAApp()
    app.run()