run: remove build _copy_directories
	docker run -d \
	-v "$$(pwd)/python/notebooks:/home/jovyan/notebooks/python:rw" \
	-v "$$(pwd)/julia/notebooks:/home/jovyan/notebooks/julia:rw" \
	-v "$$(pwd)/data:/home/jovyan/notebooks/data:rw" \
	-v "$$(pwd)/python/conda:/opt/conda:rw" \
	-v "$$(pwd)/julia/.julia:/home/jovyan/.julia:rw" \
	-p 8888:8888 \
	--name nn_python_julia \
	--user root nn_jupyter

_copy_directories:
	mkdir -p python/conda
	mkdir -p julia/.julia
	docker container stop nn_python_julia || true
	docker container rm nn_python_julia || true

	@if [ $$(find python/conda -mindepth 1 | wc -l) -eq 0 ] || [ $$(find julia/.julia -mindepth 1 | wc -l) -eq 0 ]; then \
		docker run -d \
		-v "$$(pwd)/notebooks:/notebooks:rw" \
		--user root --name nn_python_julia nn_jupyter bash; \
		docker cp nn_python_julia:/opt/conda python; \
		docker cp nn_python_julia:/home/jovyan/.julia julia; \
		docker container stop nn_python_julia; \
		docker container rm nn_python_julia; \
	fi


build:
	docker build -f Dockerfile -t nn_jupyter .

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