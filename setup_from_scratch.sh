#!/bin/bash

# programs needed to run the rest of the script
sudo apt install wget git

# Install mambaforge (conda but better, with conda-forge as the default channel)
# see https://github.com/conda-forge/miniforge#mambaforge
# and https://mamba.readthedocs.io/en/latest/installation.html
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

# Get the ICAT project
git clone https://code-int.ornl.gov/adapd/nlp/icat.git
pushd icat

# Create the conda environment for ICAT
mamba create -n icat python=3.10 jupyter jupyter-lab nodejs yarn
pip install -r requirements.txt
pip install -e .
popd

# Get the ipyanchorviz visualization project (manual dependency of icat currently)
git clone https://github.com/ORNL/ipyanchorviz
pushd ipyanchorviz

# ipyanchorviz ipywidget setup (this should be unnecessary once OSS release approved)
# This setup can also be found in the ipyanchorviz readme: https://github.com/ORNL/ipyanchorviz
pip install -e .
jupyter nbextension install --py --symlink --overwrite --sys-prefix ipyanchorviz
jupyter nbextension enable --py --sys-prefix ipyanchorviz
jupyter labextension develop --overwrite ipyanchorviz

# install the javascript dependencies for ipyanchorviz
pushd js
yarn

popd
popd
