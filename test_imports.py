# test_imports.py
import sys
from pathlib import Path

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import config...")
try:
    import config
    print(f"✓ config imported from: {config.__file__}")
    print(f"✓ EDUCATION_LEVELS: {config.EDUCATION_LEVELS}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTrying to import from agents/agent_context.py perspective...")
# Simule être dans agents/
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import EDUCATION_LEVELS
    print(f"✓ Direct import works: {EDUCATION_LEVELS}")
except Exception as e:
    print(f"✗ Direct import failed: {e}")
    # Essaye avec ..
    try:
        from .config import EDUCATION_LEVELS
        print(f"✓ Relative import works")
    except Exception as e2:
        print(f"✗ Relative import failed: {e2}")