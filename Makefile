IMAGE_NAME=facade-sam

build:
	docker build -t $(IMAGE_NAME) .

download-models:
	bash scripts/download_models.sh

run:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		$(IMAGE_NAME)

shell:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		--entrypoint /bin/bash \
		$(IMAGE_NAME)

view:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		facade-sam \
		python3 app/view_meshes.py