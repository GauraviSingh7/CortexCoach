import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.app_core import StreamlitCoachingApp

def main():
    app = StreamlitCoachingApp()
    app.run()

if __name__ == "__main__":
    main()
