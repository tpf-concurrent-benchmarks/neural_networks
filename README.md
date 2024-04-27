# Neural Networks

## How to run with Docker

### Run Jupyter in docker

````
make run
````

### Running Jupyter notebooks as a script

Make sure you have already done a `make run` and you are inside the container before running the following commands.

1. jupyter nbconvert --to script notebooks/python/training_tensorflow.ipynb
2. ipython3 notebooks/python/training_tensorflow.py

The same applies to the PyTorch notebook. 

````

### Start metrics services

```
make deploy
```


Keep in mind that the server is running on port 8888. For example, visiting `http://localhost:8888` will show the index page.

## How to run locally

````
python3 -m notebook
````