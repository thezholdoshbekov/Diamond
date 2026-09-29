"""Command-line entry points. One module per phase, invoked as `python -m diamond.cli.<phase>`.

Importing this package makes stdout/stderr UTF-8 tolerant. Without it a Windows
console (cp1251 by default) raises UnicodeEncodeError the moment a progress line
prints a non-ASCII character — and since one of those lines sits inside an
exception handler, the crash *replaces* the error it was meant to report.
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not reconfigurable (piped / captured)
        pass
