import pytest
from datetime import datetime
import torch
import psutil
import platform
from tabulate import tabulate

# Dictionary to store custom data for each test
test_data = {}

def get_system_info():
    info = {}
    info['cpu'] = platform.processor()
    info['ram'] = psutil.virtual_memory().available / (1024 ** 3)  # Available RAM in GB
    info['cuda_available'] = torch.cuda.is_available()
    if info['cuda_available']:
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_mem'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Available GPU memory in GB
    return info

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test result information and custom data."""
    if call.when == 'call':  # Only consider the 'call' stage, which is the actual test execution
        report = call.excinfo  # Access the report object
        outcome = 'passed' if report is None else ('skipped' if 'Skipped' in str(report) else 'failed')
        epoch_times = getattr(item, 'test_data', {}).get('epoch_times', {})
        inference_times = getattr(item, 'test_data', {}).get('inference_times', [])
        other_summary = getattr(item, 'test_data', {}).get('other_summary', {})
        system_info = get_system_info()
        test_data[item.nodeid] = {
            'outcome': outcome,
            'epoch_times': epoch_times,
            'inference_times': inference_times,
            'other_summary': other_summary,
            'system_info': system_info
        }

@pytest.hookimpl(tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Hook to add a summary section to the test run output."""
    terminalreporter.write_sep("=", "Detailed Test Summary")
    outcomes = {'passed': 0, 'failed': 0, 'skipped': 0}
    for data in test_data.values():
        outcomes[data['outcome']] += 1

    for outcome, count in outcomes.items():
        terminalreporter.write(f"\n{outcome.upper()}: {count} tests\n")

    terminalreporter.write_sep("=", "Custom Summary")
    for nodeid, data in test_data.items():
        terminalreporter.write(f"\nTest {nodeid}. Outcome: {data['outcome']}\n")
        system_info = data['system_info']
        if system_info:
            terminalreporter.write(f"System Info: CPU: {system_info['cpu']}, RAM: {system_info['ram']:.2f} GB, "
                                   f"CUDA Available: {system_info['cuda_available']}, "
                                   f"GPU: {system_info.get('gpu', 'N/A')}, GPU Memory: {system_info.get('gpu_mem', 'N/A')} GB\n")
        
        if data['other_summary']:
            terminalreporter.write(tabulate(data['other_summary'].items(), tablefmt="grid"))
            terminalreporter.write("\n")

        if data['epoch_times']:
            headers = ["Epochs", "Training Time (seconds)"]
            rows = [[epochs, time] for epochs, time in data['epoch_times'].items()]
            terminalreporter.write(tabulate(rows, headers=headers, tablefmt="grid"))
            terminalreporter.write("\n")

        if data['inference_times']:
            headers = ["Nr Samples", "Inference Time (seconds)"]
            rows = [[nrsamples, time] for nrsamples, time in data['inference_times'].items()]
            terminalreporter.write(tabulate(rows, headers=headers, tablefmt="grid"))
            terminalreporter.write("\n")

    # Clear the test_data dictionary for subsequent test runs
    test_data.clear()


def pytest_addoption(parser):
    parser.addoption("--modelpath", action="store", default=None, help="Path to the input file")
    parser.addoption("--shape", action="store", default=8, help="Shape of the input data")

@pytest.fixture
def filepath(request):
    modelpath = request.config.getoption("--modelpath")
    if modelpath:
        if not modelpath.endswith('.h5'):
            raise ValueError("Model file must be in h5 format")

@pytest.fixture
def shape(request):
    input_shape =  request.config.getoption("--shape")
    if input_shape:
        try:
            input_shape = int(input_shape)
        except ValueError:
            raise ValueError("Shape must be an integer")
    return input_shape


        