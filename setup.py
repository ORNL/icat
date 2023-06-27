import re

from setuptools import setup


def get_property(prop):
    result = re.search(
        rf'{prop}\s*=\s*[\'"]([^\'"]*)[\'"]', open("icat/__init__.py").read()
    )
    return result.group(1)


with open("README.md", encoding="utf-8") as infile:
    long_description = infile.read()

setup(
    name="icat-iml",
    version=get_property("__version__"),
    description="Interactive Corpus Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nathan Martindale, Scott L. Stewart",
    author_email="icat-help@ornl.gov",
    url="https://github.com/ORNL/icat",
    project_urls={"Documentation": "https://ornl.github.io/icat/"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    packages=["icat"],
    install_requires=[
        "panel",
        "numpy",
        "pandas",
        "scikit-learn",
        "ipyvuetify",
        "ipywidgets",
        "ipyanchorviz",
        "altair",
    ],
)
