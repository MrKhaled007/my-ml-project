import sys
from pathlib import Path

# Ensure `import src.<...>` resolves to this repo regardless of how pytest
# is invoked or whether `pip install -e .` registered subpackages correctly.
sys.path.insert(0, str(Path(__file__).parent))
