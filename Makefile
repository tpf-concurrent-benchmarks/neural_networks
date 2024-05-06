run: deploy build
	docker run -d \
	-v "$$(pwd)/python/notebooks:/tf/notebooks/python:rw" \
	-v "$$(pwd)/julia/notebooks:/tf/notebooks/julia:rw" \
	-v "$$(pwd)/data:/tf/notebooks/data:rw" \
	-p 8888:8888 \
	--network=nn_jupyter_metrics_default \
	--gpus all \
	--name nn_python_julia \
	--user root nn_jupyter
	docker exec nn_python_julia bash -c "./send_gpu_metrics.sh" &

deploy: remove
	until \
	docker stack deploy \
	-c docker/docker-compose.yaml \
	nn_jupyter_metrics; \
	do sleep 1; \
	done

build:
	docker build -f Dockerfile -t nn_jupyter .

remove:
	docker container stop nn_python_julia || true
	docker container rm nn_python_julia || true
	if docker stack ls | grep -q nn_jupyter_metrics; then \
		docker stack rm nn_jupyter_metrics; \
	fi

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