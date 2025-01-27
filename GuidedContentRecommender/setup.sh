
echo "Installing requirements..."
pip3 install -r requirements.txt
brew install libomp
brew install pandoc librsvg
jupyter labextension install jupyterlab-plotly

echo "init pre-commit hooks"
pre-commit install
