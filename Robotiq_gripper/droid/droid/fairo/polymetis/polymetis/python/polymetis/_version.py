import os
import json

import polymetis

__version__ = ""

# Conda installed: Get version of conda pkg (assigned $GIT_DESCRIBE_NUMBER during build)
if "CONDA_PREFIX" in os.environ and os.environ["CONDA_PREFIX"] in polymetis.__file__:
    # Search conda pkgs for polymetis & extract version number
    stream = os.popen("conda list | grep polymetis")
    for line in stream:
        info_fields = [s for s in line.strip("\n").split(" ") if len(s) > 0]
        if info_fields[0] == "polymetis":  # pkg name == polymetis
            __version__ = info_fields[1]
            break

# Built locally: Retrive git tag description of Polymetis source code
else:
    try:
        original_cwd = os.getcwd()
        os.chdir(os.path.dirname(polymetis.__file__))

        stream = os.popen("git describe --tags")
        lines = [line for line in stream]

        if len(lines) > 0:
            version_string = lines[0]
            version_items = version_string.strip("\n").split("-")
            __version__ = f"{version_items[-2]}_{version_items[-1]}"
        else:
            __version__ = "0.2"

        os.chdir(original_cwd)

    except Exception:
        __version__ = "0.2"
        
if not __version__:
    raise Exception("Cannot locate Polymetis version!")
