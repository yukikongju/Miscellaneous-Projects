
echo "Installing requirements..."
pip3 install -r requirements.txt

echo "init pre-commit hooks"
pre-commit install
