#### Running Tests

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing. 

- To run the tests, navigate to the mini-test directory
```bash
cd mini-test
```

- Run the following command in the root directory of the project to run all the tests:
```bash
pytest
```

- To run a specific test file, you can run the following command:
```bash
pytest tests/test_file.py
```
 Don't forget to replace `test_file.py` with the name of the test file you want to run.

- To run a specific test function, you can run the following command:
```bash
pytest tests/test_file.py::test_function
```

Run faultid model test
```bash
pytest test_model.py --modelpath {specify faultmodel path} --shape {specify input shape}
```

- Example:
```bash
pytest test_model.py --modelpath models/faultid_model.h5 --shape 128
```