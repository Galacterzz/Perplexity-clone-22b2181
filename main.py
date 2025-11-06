# main.py
import sys
import os

# Add the project root directory to the Python path
# This allows imports like 'from src.ui import main_ui' to work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui import main_ui

if __name__ == "__main__":
    # Call the main UI function defined in src.ui
    main_ui()