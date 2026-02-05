import sys
import os
from streamlit.web import cli as stcli

def main():
    sys.argv = ["streamlit", "run", "app/ui/streamlit_app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
