#!/bin/bash

sudo apt-get install -y gfortran libopenblas-dev liblapack-dev libfreetype6-dev libpng-dev libjpeg-dev pkg-config libncurses5-dev python-tk python-pip

pip install --upgrade --user Cython Jinja2 MarkupSafe Pillow Pygments appnope argparse backports-abc backports.ssl-match-hostname certifi cycler decorator future gnureadline ipykernel ipython ipython-genutils ipywidgets jsonschema jupyter jupyter-client jupyter-console jupyter-core matplotlib mistune nbconvert nbformat notebook numpy path.py pexpect pickleshare ptyprocess pyparsing python-dateutil pytz pyzmq qtconsole scipy simplegeneric singledispatch site six terminado tornado traitlets librosa

