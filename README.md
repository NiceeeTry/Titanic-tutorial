[![Tests are](https://github.com/NiceeeTry/titanic-tutorial/actions/workflows/python-app.yml/badge.svg)](https://github.com/NiceeeTry/titanic-tutorial/actions/workflows/python-app.yml)
## Usage
```
Usage: main.py [OPTIONS] INPUT_PATH TEST_PATH

Arguments:
  INPUT_PATH  [required]
  TEST_PATH   [required]

Options:
  --result-path TEXT  [default: data/predictions.csv]
  --help              Show this message and exit.
```

If you are in the root directory, run the following command:
```
poetry run .\src\titanic_tutorial\main.py "data/train.csv" "data/test.csv"
```
The result of the program is the `predictions.csv` file located in `data` directory.
