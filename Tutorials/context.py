from pathlib import Path
import sys
path_root = Path("context.py").resolve()
path_root = path_root.parents[1] / "src"
sys.path.append(str(path_root))
from DSGRN_sheaves import *