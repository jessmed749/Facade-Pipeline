IMAGE_NAME=facade-sam
USER_FLAG=--user $(shell id -u):$(shell id -g)

build:
	docker build -t $(IMAGE_NAME) .

download-models:
	bash scripts/download_models.sh

run-pipeline: facade geometry simulate

facade:
	@echo "\nExtracting facades (Linux)..."
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-v $(PWD)/.hf_cache:/root/.cache/huggingface \
		$(IMAGE_NAME) python3 app/run_facade_pipeline.py

geometry:
	@echo "\nRunning ArcGIS Digital Twin Assembly (Windows)..."
	powershell.exe -Command "New-Item -ItemType Directory -Force -Path 'automated_digital_twin/source_data' | Out-Null"
	powershell.exe -Command "Copy-Item -Path 'output/combined_scene.obj' -Destination 'automated_digital_twin/source_data/' -Force"
	powershell.exe -Command "Copy-Item -Path 'output/facade_combined_scene.json' -Destination 'automated_digital_twin/source_data/' -Force"
	@echo 'call "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\Scripts\\activate.bat" "%USERPROFILE%\\AppData\\Local\\ESRI\\conda\\envs\\campus_twin_env"' > run_arcgis_temp.bat
	@echo 'python automated_digital_twin/main.py' >> run_arcgis_temp.bat
	cmd.exe /c run_arcgis_temp.bat
	@rm run_arcgis_temp.bat

# Anchor facades to a previously generated digital twin, requires the data.gdb created from the geometry command.
anchor-facades-only:
	@echo "\nAnchoring Facades to Digital Twin (Windows)..."
	@echo 'call "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\Scripts\\activate.bat" "%USERPROFILE%\\AppData\\Local\\ESRI\\conda\\envs\\campus_twin_env"' > run_arcgis_temp.bat
	@echo 'set PYTHONPATH=automated_digital_twin' >> run_arcgis_temp.bat
	@echo 'python -c "from automated_digital_twin import main; main.anchor_facades()"' >> run_arcgis_temp.bat
	cmd.exe /c run_arcgis_temp.bat
	@rm run_arcgis_temp.bat

simulate:
	@echo "\nRunning Sionna simulations..."
	docker run --rm -it --gpus all \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-e FACADE_OUTPUT_DIR=/workspace/output \
		$(IMAGE_NAME) python3 app/sionna_scene_loader.py
	docker run --rm -it --gpus all \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		$(IMAGE_NAME) python3 app/run_sionna.py

shell:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-v $(PWD)/.hf_cache:/root/.cache/huggingface \
		-v /usr/lib/wsl:/usr/lib/wsl \
		-e DRJIT_LIBOPTIX_PATH=/usr/lib/wsl/lib \
		-e LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH \
		--entrypoint /bin/bash \
		$(IMAGE_NAME)
		
view:
	docker run --rm -it --gpus all \
		-v $(PWD)/checkpoints:/workspace/checkpoints \
		-v $(PWD)/input:/workspace/input \
		-v $(PWD)/output:/workspace/output \
		-v $(PWD)/app:/workspace/app \
		-v $(PWD)/.hf_cache:/root/.cache/huggingface \
		facade-sam \
		python3 app/view_meshes.py

clean:
	sudo rm -rf output/masks/* output/meshes/* output/per_class/* \
		output/combined_scene.obj output/sionna_scene.json \
		output/mask_overlay.png
	@echo "output cleared"