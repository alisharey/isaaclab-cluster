#!/usr/bin/env python3

"""Run pip check while narrowly recognizing one upstream version conflict."""

from __future__ import annotations

import re
import subprocess
import sys


KNOWN_CONFLICT = re.compile(
    r"^isaaclab 0\.54\.2 has requirement starlette==0\.49\.1, "
    r"but you have starlette 0\.45\.3\.$"
)


result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    capture_output=True,
    text=True,
)
lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]

for line in lines:
    print(line)

if result.returncode == 0:
    print("PIP_DEPENDENCIES_OK")
elif len(lines) == 1 and KNOWN_CONFLICT.fullmatch(lines[0]):
    print("PIP_DEPENDENCIES_OK_WITH_DOCUMENTED_STARLETTE_CONFLICT")
else:
    print("Unexpected Python dependency conflict(s).", file=sys.stderr)
    raise SystemExit(result.returncode or 1)
