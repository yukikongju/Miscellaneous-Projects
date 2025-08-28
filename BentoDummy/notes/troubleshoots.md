# Troubleshoot

- `UnboundLocalError: local variable 'local_path' referenced before assignment` => need to set `mlflow.set_uri()` before
- `error: RPC failed; HTTP 400 curl 22 The requested URL returned error: 400` => `git config http.postBuffer 524288000; git pull && git push`
- `ERROR: (gcloud.run.deploy) Revision 'iris-inference-service-00001-b89' is not ready and cannot serve traffic. Cloud Run does not support image 'gcr.io/relax-server/iris_classifier:cb3l6duee2nq3zv2': Container manifest type 'application/vnd.oci.image.index.v1+json' must support amd64/linux.` =>
