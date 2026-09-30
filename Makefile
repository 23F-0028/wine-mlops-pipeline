.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	python -m flake8 src/ tests/ --max-line-length=100

test:
	python -m pytest -v

train:
	python src/train.py

clean:
	powershell -Command "Get-ChildItem -Path . -Recurse -Filter *.pyc -File | Remove-Item -Force"
	powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force"
	powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force"
	powershell -Command "Get-ChildItem -Path . -Recurse -Include *.tmp,*.temp -File | Remove-Item -Force"