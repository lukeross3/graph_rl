# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = ""

setup(
    long_description=readme,
    name="graph_rl",
    version="0.1.0",
    description="RL-based runtime optimization of computation graphs",
    python_requires="==3.*,>=3.6.0",
    author="Luke Ross",
    packages=["graph_rl"],
    package_dir={"": "."},
    package_data={},
    install_requires=["mpi4py==3.*,>=3.0.3", "numpy==1.*,>=1.19.4", "toml==0.*,>=0.10.2"],
    extras_require={
        "dev": [
            "black==20.*,>=20.8.0.b1",
            "coverage==5.*,>=5.3.1",
            "dephell==0.*,>=0.8.3",
            "pytest==5.*,>=5.2.0",
            "pytest-mock==3.*,>=3.4.0",
        ]
    },
)
