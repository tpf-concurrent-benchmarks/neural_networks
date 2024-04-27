run: remove build _copy_directories
	docker run -d \
	-v "$$(pwd)/python/notebooks:/tf/notebooks/python:rw" \
	-v "$$(pwd)/julia/notebooks:/tf/notebooks/julia:rw" \
	-v "$$(pwd)/data:/tf/notebooks/data:rw" \
	-v "$$(pwd)/python/dist-packages:/usr/local/lib/python3.11/dist-packages:rw" \
	-v "$$(pwd)/julia/.julia:/root/.julia:rw" \
	-p 8888:8888 \
	--gpus all \
	--name nn_python_julia \
	--user root nn_jupyter

run_tensorflow:
	docker exec -it $$(make run | tail -1) bash
	jupyter nbconvert --to script notebooks/python/training_tensorflow.ipynb
	ipython3 notebooks/python/training_tensorflow.py

run_pytorch:
	docker exec -it $$(make run | tail -1) bash
	jupyter nbconvert --to script notebooks/python/training_pytorch.ipynb
	ipython3 notebooks/python/training_pytorch.py

deploy: remove_stack
	until \
	docker stack deploy \
	-c docker/docker-compose.yaml \
	nn_jupyter_metrics; \
	do sleep 1; \
	done

remove_stack:
	if docker stack ls | grep -q nn_jupyter_metrics; then \
		docker stack rm nn_jupyter_metrics; \
	fi

_copy_directories:
	mkdir -p graphite
	mkdir -p python/dist-packages
	mkdir -p julia/.julia
	docker container stop nn_python_julia || true
	docker container rm nn_python_julia || true

	@if [ $$(find python/dist-packages -mindepth 1 | wc -l) -eq 0 ] || [ $$(find julia/.julia -mindepth 1 | wc -l) -eq 0 ]; then \
		docker run -d \
		-v "$$(pwd)/notebooks:/notebooks:rw" \
		--user root --name nn_python_julia nn_jupyter bash; \
		docker cp nn_python_julia:/usr/local/lib/python3.11/dist-packages python; \
		docker cp nn_python_julia:/root/.julia julia; \
		docker container stop nn_python_julia; \
		docker container rm nn_python_julia; \
	fi


build:
	docker build -f Dockerfile -t nn_jupyter .

build_julia:
	docker build -f Dockerfile-julia -t nn_jupyter .

remove:
	docker container stop nn_python_julia || true
	docker container rm nn_python_julia || true

logs:
	docker container logs nn_python_julia

# You need the container to be running and the following
# libraries installed *in the conda environment of the container*:
# - pandas
# - sklearn
datasets:
	docker exec nn_python_julia bash -c "cd notebooks/data && unzip -n archive.zip && python3 train_test_val_generator.py"

#returns the link of the server
get_link:
	docker exec -it nn_python_julia jupyter server list