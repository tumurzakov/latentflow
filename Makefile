include env
export

unit:
	cd tests/unit; PYTHONPATH=../../src python -m unittest *_test.py

func:
	cd tests/func; PYTHONPATH="../../src:$(ANIMATEDIFF_PATH):$(COMFYUI_PATH)" python -m unittest test_*.py
