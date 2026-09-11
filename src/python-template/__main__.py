"""Allow the package to run with ``python -m python_template``."""

from python_template.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
