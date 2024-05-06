# Neural Networks - Python and Julia

## Objective

These are a Python and Julia implementations of a neural network for performing training and prediction under [common specifications](https://github.com/tpf-concurrent-benchmarks/docs/tree/main/neural_network) defined for multiple languages.

The objective of this project is to benchmark both languages on a common task, and to compare the performance of the neural network implementations in both languages.

## Deployment

### Requirements

- [Docker >3](https://www.docker.com/) (needs docker swarm)
- [Julia](https://julialang.org/downloads/) (for local builds)
- [Python 3.10](https://www.python.org/downloads/) (for local builds)

### Configuration

- **Dataset:** in `data/archive.zip` you can find the dataset used for training and testing the neural network.

### Commands

#### Startup

- `make build` will build the docker image used in both languages.

#### Run

- `make run` will run the system. The notebooks will be available at [http://localhost:8888](http://localhost:8888).
- `make remove` removes all services.
- `make datasets` will split the dataset into training, testing and validation sets.

Additionally, the notebook `training_pytorch.ipynb` must be executed by a Python 3.10 interactive kernel which should be already available into the container.

### Monitoring

- Grafana: [http://127.0.0.1:8081](http://127.0.0.1:8081)
- Graphite: [http://127.0.0.1:8080](http://127.0.0.1:8080)
- Logs

## Libraries

- [PyTorch](https://pytorch.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Scikeras](https://adriangb.com/scikeras/stable/)
- [ScikitLearn](https://cstjean.github.io/ScikitLearn.jl/dev/)
- [Flux.jl](https://fluxml.ai/Flux.jl/stable/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

