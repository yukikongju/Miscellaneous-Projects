dockerize:
	# --- containerize image
	bentoml list
	bentoml build
	bentoml containerize --platform linux/amd64 iris_classifier:latest
	docker run --rm -p 3000:3000 iris_classifier:cb3l6duee2nq3zv2
	# --- to push image to gcp container registry
	docker tag iris_classifier:cb3l6duee2nq3zv2 gcr.io/relax-server/iris_classifier:cb3l6duee2nq3zv2
	docker push gcr.io/relax-server/iris_classifier:cb3l6duee2nq3zv2
	# --- deploy to cloud run for inference
	# - not enough permission: 'run.services.setIamPolicy'
	gcloud beta run services add-iam-policy-binding --region=us-central1 --member=allUsers --role=roles/run.invoker iris-inference-service
	#
	gcloud run deploy iris-inference-service \
		--image gcr.io/relax-server/iris_classifier:latest \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated

inference:
	bentoml serve
	curl -X 'POST' \
	  'http://localhost:3000/predict' \
	  -H 'accept: application/json' \
	  -H 'Content-Type: application/json' \
	  -d '{
	  "input_data": [[
	    0.1, 0.2, 0.1, 0.1
	  ]]
	}'
