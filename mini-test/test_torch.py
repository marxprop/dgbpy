import pytest

import sys
sys.path.insert(0, '..')

import pytest
from functools import partial
import dgbpy.keystr as dbk
import dgbpy.dgbtorch as dgbtorch
import dgbpy.torch_classes as tc
import torch
import torch.nn as nn
import time
import torch
import gc

from init_data import *

def default_pars():
    pars = dgbtorch.torch_dict.copy()
    pars['epochs'] = 15
    pars['batch'] = 2
    pars[dbk.prefercpustr] = True
    pars['tofp16'] = False
    pars['nbfold'] = 1
    return pars


def get_default_model(info):
    learntype, classification, nrdims, nrattribs = getExampleInfos(info)
    modeltype = dgbtorch.getModelsByType(learntype, classification, nrdims)
    return modeltype


def get_model_arch(info, model, model_id):
    architecture = dgbtorch.getDefaultModel(info, type=model[model_id])
    return architecture


def train_model(trainpars=default_pars(), data=None):
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    start_time = time.time()
    model = dgbtorch.train(modelarch, data, trainpars)
    end_time = time.time()
    duration = end_time - start_time

    # Clear GPU memory and force garbage collection
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    return model, info, duration

def apply_model(model=None, data=None, nrsamples = None):
    info = data[dbk.infodictstr]

    if not nrsamples:
        samples = data[dbk.xvaliddictstr]
    else:
        samples = np.random.rand(nrsamples, *data[dbk.xvaliddictstr].shape[1:])

    isclassification = info[dbk.classdictstr]
    withpred = True
    withprobs = []
    withconfidence = False
    doprobabilities = len(withprobs) > 0

    start_time = time.time()
    dgbtorch.apply(model, info, samples, None, isclassification, withpred, withprobs, withconfidence, doprobabilities)
    end_time = time.time()
    duration = end_time - start_time
    gc.collect()
    return duration

def save_model(model, filename, info, params):
    dgbtorch.save(model, filename, info, params)
    return model, filename, info


def load_model(filename, info=None):
    model = dgbtorch.load(filename, info)
    return model

def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ================================================ TESTS ================================================
@pytest.fixture
def test_data(request):
    """Fixture to store custom test data."""
    return {}

def test_2d_imgtoimg(request, test_data):
    """Test for 2D image-to-image model training."""
    inpshape = [1, 8, 8]
    outshape = [1, 8, 8]
    epoch_trials = list(range(5, 21, 5))
    inference_trials = [1, 8, 64, 128]
    nrpts = 16

    data = get_2d_seismic_imgtoimg_data(nrpts = nrpts, inpshape=inpshape, outshape=outshape)
    train_summary, inference_summary, other_summary = {},{},{}

    for epochs in epoch_trials:
        train_pars = default_pars()
        train_pars['epochs'] = epochs
        model, _, duration = train_model(data=data, trainpars=train_pars)
        train_summary[epochs] = duration

    test_data['epoch_times'] = train_summary

    for nrsamples in inference_trials:
        duration = apply_model(model=model, data=data, nrsamples=nrsamples)
        inference_summary[nrsamples] = duration

    test_data['inference_times'] = inference_summary

    total_params, trainable_params = count_parameters(model)

    #0ther summary
    other_summary['Platform'] = 'Pytorch'
    other_summary['Input Shape'] = inpshape
    other_summary['Output Shape'] = outshape
    other_summary['Total Model Parameters'] = total_params
    other_summary['Trainable Model Parameters'] = trainable_params

    test_data['other_summary'] = other_summary

    request.node.test_data = test_data


def test_3d_imgtoimg(request, test_data):
    """Test for 3D image-to-image model training."""
    inpshape = [16, 16, 16]
    outshape = [16, 16, 16]
    epoch_trials = list(range(5, 21, 5))
    inference_trials = [1, 8, 64, 128]
    nrpts = 32

    data = get_3d_seismic_imgtoimg_data(nrpts = nrpts, inpshape=inpshape, outshape=outshape)
    train_summary, inference_summary, other_summary = {},{},{}

    for epochs in epoch_trials:
        train_pars = default_pars()
        train_pars['epochs'] = epochs
        model, _, duration = train_model(data=data)
        train_summary[epochs] = duration

    test_data['epoch_times'] = train_summary

    for nrsamples in inference_trials:
        duration = apply_model(model=model, data=data, nrsamples=nrsamples)
        inference_summary[nrsamples] = duration

    test_data['inference_times'] = inference_summary

    total_params, trainable_params = count_parameters(model)

    #0ther summary
    other_summary['Platform'] = 'Pytorch'
    other_summary['Input Shape'] = inpshape
    other_summary['Output Shape'] = outshape
    other_summary['Total Model Parameters'] = total_params
    other_summary['Trainable Model Parameters'] = trainable_params

    test_data['other_summary'] = other_summary

    request.node.test_data = test_data


# - Set input size to 96
# - Give option to test training/inference separately
# - Overall runtime of applying the model(faultid) should be computed
# - Mention that the input is random and the test doesnt include IO time from reading the data (It can be significantly larger in real world)
# - Allows us to understand the runtime on different architectures.
# - Checkout aws graphiton for ML 
# - https://github.com/aws/aws-graviton-getting-started/tree/main/machinelearning
