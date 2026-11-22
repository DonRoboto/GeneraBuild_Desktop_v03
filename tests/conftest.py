import sys
from pathlib import Path
TESTS=Path(__file__).parent
for path in (TESTS,TESTS.parent):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
