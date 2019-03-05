#!/bin/bash --login

source /root/.bashrc
conda activate gluon
/opt/conda/bin/jupyter notebook --notebook-dir=/d2l-en --ip='0.0.0.0' --port=8888 --no-browser --allow-root --NotebookApp.token='d2l'
