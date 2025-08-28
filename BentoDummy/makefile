dockerize:
	bentoml build
	bentoml containerize iris_classifier:cb3l6duee2nq3zv2
	docker run --rm -p 3000:3000 iris_classifier:cb3l6duee2nq3zv2
