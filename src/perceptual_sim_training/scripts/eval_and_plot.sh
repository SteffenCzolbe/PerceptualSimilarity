# convert jupyter notebook to python and run it
jupyter nbconvert --to python eval.ipynb

# evaluate all trained models
python eval.py

# convert jupyter notebook to python and run it
jupyter nbconvert --to python eval-plot.ipynb

# evaluate all trained models
python eval-plot.py