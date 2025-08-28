BENTO_MODEL_NAME=embedding
BENTO_BUILD_NAME=guided_content_embedding

list_bento_build:
	bentoml list

containerize_bento:
	# docs: https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html
	bentoml build
	# bentoml containerize $(BENTO_BUILD_NAME):latest
	bentoml containerize --platform=linux/amd64 $(BENTO_BUILD_NAME):latest
	docker images
	# docker run --rm -p 3000:3000 guided_content_embedding:qec2e2ud5s2slzv2
	# docker run -it --rm -p 3000:3000 -v ~/bentoml:/home/bentoml guided_content_embedding:s4fgt5edvcsehzv2 serve
	# docker run -it --rm -p 3000:3000 guided_content_embedding:s4fgt5edvcsehzv2 serve

bento_inference:
	bentoml serve
	python3 inference.py


cleanup_bento_models:
	# remove bento model except the latest one
	bentoml models list | grep $(BENTO_MODEL_NAME) | tail -n +2 | awk '{print $1}' | xargs -r -n1 bentoml models delete -y


bento_api:
	http://0.0.0.0:3000/

bentoml_model:
	bentoml models list

bento_serve:
	# when service.py is defined
	bentoml serve

# similar:
	# {
	#   "input_data": [
	#     "word", "new"
	#   ],
	#   "top_n": 10
	# }
