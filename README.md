# Neural Networks

## How to run with Docker

### Run Jupyter in docker

````
make run
````

### Start metrics services

```
make deploy
```


Keep in mind that the server is running on port 8888. For example, visiting `http://localhost:8888` will show the index page.

Additionally, the notebook `training_pytorch.ipynb` must be executed by a Python 3.10.12 interactive kernel which should be already available into the container.
