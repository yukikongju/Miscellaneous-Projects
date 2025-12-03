open_jupyter:
	uv run jupyter notebook
	uv run jupyter lab

uv:
	uv cache clean
	uv sync --refresh

optimum_cli:
	optimum-cli export onnx \
	  --model sentence-transformers/all-MiniLM-L6-v2 \
	  onnx_model/
