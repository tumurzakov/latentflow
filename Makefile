include env
export

unit:
	cd src/tests_unit; PYTHONPATH=.. python -m unittest *_test.py

func:
	cd src/tests_func; PYTHONPATH="..:../../../AnimateDiff" python -m unittest test_*.py
