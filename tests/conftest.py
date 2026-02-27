import pytest
import sys
from pathlib import Path
from dotenv import load_dotenv

tests_dir = Path(__file__).resolve().parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
