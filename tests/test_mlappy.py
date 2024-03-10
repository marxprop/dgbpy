import sys
sys.path.insert(0, '..')

import os
import fnmatch, shutil, copy, pytest
from functools import partial
import dgbpy.keystr as dbk
import dgbpy.mlapply as dgbml
import dgbpy.hdf5 as dgbhdf5
from init_data import *
from test_dgkeras import default_pars as keras_params
from test_dgbscikit import default_pars as scikit_params
from test_dgbtorch import default_pars as torch_params

def get_example_files():
    storageFolder = os.path.join(os.getcwd(), 'tests/test_data')
    examplefilenm = []
    for file in os.listdir(storageFolder):
        if fnmatch.fnmatch(file, '*.h5'):
            examplefilenm.append(os.path.join(storageFolder, file))
    return examplefilenm

test_data_ids = []
examples = get_example_files()

@pytest.mark.parametrize('examplefilenm', [examples[0]])
def test_doTrain_invalid_platform(examplefilenm, capsys):
    kwargs = {
        'platform': 'invalid_platform',
        'type': dgbml.TrainType.New,
        'params': {},
        'outnm': 'invalid_platform_test',
        'logdir': None,
        'clearlogs': False,
        'modelin': None,
    }
    with pytest.raises(AttributeError):
        dgbml.doTrain(examplefilenm, **kwargs)
        assert dgbml.doTrain('test_data/invalid_platform_test.h5', **kwargs) == False
        captured = capsys.readouterr()
        assert 'Unsupported machine learning platform' in captured.out

def keras_test_cases():
    params = keras_params()
    return {
        'platform': dbk.kerasplfnm,
        'type': dgbml.TrainType.New,
        'params': params,
        'outnm': 'keras_test',
        'logdir': None,
        'clearlogs': False,
        'modelin': None,
    }

@pytest.mark.parametrize('examplefilenm', examples)
def test_doTrain_keras_new_trainingtype(examplefilenm):
    kwargs = keras_test_cases()
    assert dgbml.doTrain(examplefilenm, **kwargs) == True

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_keras_resume_trainingtype(examplefilenm):
#     kwargs = keras_test_cases()
#     kwargs['type'] = dgbml.TrainType.Resume
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_keras_transfer_trainingtype(examplefilenm):
#     kwargs = keras_test_cases()
#     kwargs['type'] = dgbml.TrainType.Transfer
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# def torch_test_cases():
#     params = torch_params()
#     return {
#         'platform': dbk.torchplfnm,
#         'type': dgbml.TrainType.New,
#         'params': params,
#         'outnm': 'torch_test',
#         'logdir': None,
#         'clearlogs': False,
#         'modelin': None,
#     }

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_torch_new_trainingtype(examplefilenm):
#     kwargs = torch_test_cases()
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_torch_resume_trainingtype(examplefilenm):
#     kwargs = torch_test_cases()
#     kwargs['type'] = dgbml.TrainType.Resume
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_torch_transfer_trainingtype(examplefilenm):
#     kwargs = torch_test_cases()
#     kwargs['type'] = dgbml.TrainType.Transfer
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# def scikit_test_cases():
#     params = scikit_params()
#     return {
#         'platform': dbk.scikitplfnm,
#         'type': dgbml.TrainType.New,
#         'params': params,
#         'outnm': 'scikit_test',
#         'logdir': None,
#         'clearlogs': False,
#         'modelin': None,
#     }

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_scikit_new_trainingtype(examplefilenm):
#     kwargs = scikit_test_cases()
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_scikit_resume_trainingtype(examplefilenm):
#     kwargs = scikit_test_cases()
#     kwargs['type'] = dgbml.TrainType.Resume
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True

# @pytest.mark.parametrize('examplefilenm', examples)
# def test_doTrain_scikit_transfer_trainingtype(examplefilenm):
#     kwargs = scikit_test_cases()
#     kwargs['type'] = dgbml.TrainType.Transfer
#     assert dgbml.doTrain(examplefilenm, **kwargs) == True
