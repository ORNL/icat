import re

from setuptools import setup


def get_property(prop):
    result = re.search(
        rf'{prop}\s*=\s*[\'"]([^\'"]*)[\'"]', open("icat/__init__.py").read()
    )
    return result.group(1)


with open("README.md", encoding="utf-8") as infile:
    # TODO: will have to remove the <picture><source ...</picture>
    # since pypi doesn't support
    long_description = infile.read()

    # remove non-pypi friendly picture/source tags
    lines = long_description.split("\n")
    del lines[5]
    del lines[3]
    del lines[2]
    del lines[1]
    long_description = "\n".join(lines)

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
    package_data={"icat": ["vue/*"]},
    install_requires=[
        "panel",
        "pyviz_comms",
        "numpy",
        "pandas",
        "scikit-learn",
        "ipyanchorviz",
        "altair",
        "jupyter_bokeh",
        "ipywidgets-bokeh",
        "ipyvuetify~=1.9",
        "ipyvue~=1.11",
        "ipywidgets<8.1.3",  # can be removed once https://github.com/holoviz/panel/issues/6921 is resolved (panel 1.5?)
        "jupyterlab-widgets<3.0.11",  # "
    ],
    # an exact set of dependencies that I know for a fact works, remove comments
    # in case of emergency
    # install_requires=[
    #     "panel<1.2",
    #     "pyviz_comms<3",
    #     "numpy",
    #     "pandas",
    #     "scikit-learn",
    #     "ipyanchorviz",
    #     "altair",
    #     "ipyvue==1.9.1",
    #     "ipyvuetify==1.8.10",
    #     "ipywidgets==8.0.6",
    #     "param<2",
    #     "traitlets==5.9.0",
    #     "bokeh==3.1.1",
    #     "jupyterlab==4.1.2",
    #     "jupyterlab_server==2.23.0",
    #     "jupyterlab-widgets==3.0.7",
    #     "ipywidgets-bokeh",
    #     "jupyter_bokeh",
    # ],
)
