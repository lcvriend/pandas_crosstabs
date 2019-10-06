#! /bin/bash
conda env create -f environment.yml
conda activate pandas_crosstabs
python -m ipykernel install --user --name pandas_crosstabs --display-name pandas_crosstabs
