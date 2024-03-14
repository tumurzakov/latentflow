include env
export

PYTHON_UNIT=cd tests/unit; PYTHONPATH=../../src python
PYTHON_FUNC=cd tests/func; PYTHONPATH="../../src:$(ANIMATEDIFF_PATH):$(COMFYUI_PATH)" python

unit:
	$(PYTHON_UNIT) -m unittest *_test.py

func:
	$(PYTHON_FUNC) -m unittest test_*.py

func_single:
	$(PYTHON_FUNC) -m unittest $(FILE) -v; \
