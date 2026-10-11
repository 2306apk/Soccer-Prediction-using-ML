import sys
from pathlib import Path

# Ensure project root is in sys.path for all pytest invocations
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
