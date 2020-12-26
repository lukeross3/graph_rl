import os
import re
import toml

# Read the pyproject version
pyproj_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "pyproject.toml"
)
pyproj = toml.load(pyproj_path)
version = pyproj["tool"]["poetry"]["version"]

# Read the __init__.py file into a string
init_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "graph_rl", "__init__.py"
)
with open(init_path, "r") as f:
    init_str = f.read()

# Replace the __init__ version with the pyproject version
init_str = re.sub('__version__\s*=\s*"\d+\.\d+\.\d+"', f'__version__ = "{version}"', init_str)

# Write new string to __init__.py
with open(init_path, "w") as f:
    f.write(init_str)
