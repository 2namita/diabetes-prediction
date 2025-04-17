#Makefile
install:
	pip install -r requirements/test_requirements.txt
format:
	
lint:

clean:
	pip uninstall diabetes_model
	
build: 	
	python3 diabetes_model/train_pipeline.py --force

test:
	pytest 

package:
	pip install --upgrade build
	python3 -m build
	
all:clean install  build test package 