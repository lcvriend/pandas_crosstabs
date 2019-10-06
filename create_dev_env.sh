#! /bin/bash
conda env create -f environment.yml
conda activate pandas-glit
python -m ipykernel install --user --name pandas-glit --display-name pandas-glit
