import setuptools

long_description = """
To be very honest, I don't yet understand this project enough to write a long description. 
"""

setuptools.setup(
    name="nanoporeter",
    version="0.0.1",
    author="MISL",
    #package_dir={'': 'src'},
    #author_email="author@example.com",
    description="Using nanopores to...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uwmisl",
    packages=setuptools.find_packages(),
    provides=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=2.7",
    setup_requires = ["pytest"]
)
