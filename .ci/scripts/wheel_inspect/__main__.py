"""Entry point for ``python3 .ci/scripts/wheel_inspect``."""

import sys
from pathlib import Path

# Running the directory puts it, not its parent, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from wheel_inspect.cli import main

    sys.exit(main())
