### __setup__.py
### MIT LICENSE 2024 Alex Dowling

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DSGRN_sheaves",
    version="0.0.1",
    author="Alex Dowling",
    author_email="alex@dowlinghome.com",
    description="Implementation of a sheaf data structure for DSGRN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexDowling/DSGRN_sheaves",
    package_dir={'':'src'},
    packages=['DSGRN_sheaves'],
    install_requires=["DSGRN", "pyCHomP2", "DSGRN_utils", "galois"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
