# Deployment

Model Deployment will be done with:
- MLFlow: save artifacts and logs
- BentoML: provide API endpoint to serve model



What needs to be done:
- [X] Save model in MLFlow
- [X] Create BentoML function for (1) vectorize content (2) get closest
- [X] Create Docker Image
- [ ] Deploy Docker Image 
- [ ] Make inference from Docker
- [ ] Bonus: BentoCloud


**Troubleshoot**
- `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?` => need to run Docker Desktop
- `Error: [bentoml-cli] containerize failed: no Bentos with name 'embedding' exist in BentoML store /Users/emulie/bentoml/bentos`
- `FileNotFoundError: [Errno 2] No such file or directory: '/home/bentoml/models/embedding/latest'; bentoml.exceptions.NotFound: no Models with name 'embedding' exist in BentoML store /home/bentoml/models` => 
`
`


