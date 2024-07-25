import sys
sys.path.insert(0, '..')

import os, pytest, time
import dgbpy.keystr as dbk
import dgbpy.dgbkeras as dgbkeras
import dgbpy.hdf5 as dgbhdf5
import dgbpy.keras_classes as kc
import keras
from init_data import *

all_data = lambda **kwargs: (
    get_2d_seismic_imgtoimg_data(**kwargs),
    get_3d_seismic_imgtoimg_data(**kwargs),
    get_seismic_classification_data(**kwargs),
    get_loglog_data(**kwargs),
    get_loglog_classification_data(**kwargs),
)

test_data_ids = ['2D_seismic_imgtoimg', '3D_seismic_imgto_img', 'seismic_classification', 'loglog_regression', 'log_classification']


def default_pars():
    pars = dgbkeras.keras_dict
    pars['epochs'] = 10
    pars['batch'] = 4
    pars[dbk.prefercpustr] = True
    pars['tofp16'] = False
    return pars


def get_default_model(info):
    learntype, classification, nrdims, nrattribs = getExampleInfos(info)
    modeltypes = dgbkeras.getModelsByType(learntype, classification, nrdims)
    return modeltypes


def get_model_arch(info, model, model_id):
    architecture = dgbkeras.getDefaultModel(info, type=model[model_id])
    return architecture

def is_model_trained(initial, current):
    for initial_layer, current_layer in zip(initial, current):
        if not np.array_equal(initial_layer, current_layer):
            return True 
    return False

def save_model(model, filename):
    dgbkeras.save(model, filename)
    return model, filename

def load_model(filename):
    return dgbkeras.load(filename, False)

def train_model(trainpars=default_pars(), data=None):
    info = data[dbk.infodictstr]
    model = get_default_model(info)
    modelarch = get_model_arch(info, model, 0)
    start_time = time.time()
    model = dgbkeras.train(modelarch, data, trainpars, silent=True)
    end_time = time.time()
    duration = end_time - start_time
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

    inpshape = info[dbk.inpshapedictstr]
    if isinstance(inpshape, int):
        inpshape = [inpshape]
    dictinpshape = tuple( inpshape )

    start_time = time.time()
    dgbkeras.apply(model, info, samples, isclassification, withpred=withpred, withprobs=withprobs, withconfidence=withconfidence, doprobabilities=doprobabilities, \
                                dictinpshape=dictinpshape, scaler=None, batch_size=4)
    end_time = time.time()
    duration = end_time - start_time
    return duration

@pytest.fixture
def test_data(request):
    """Fixture to store custom test data."""
    return {}

def test_2d_seismic_imgtoimg(request, test_data):
    """Test for 2D image-to-image model training."""
    inpshape = [1, 16, 16]
    outshape = [1, 16, 16]
    epoch_trials = list(range(5, 21, 5))
    inference_trials = [1, 8, 64, 128]
    nrpts = 32

    data = get_2d_seismic_imgtoimg_data(nrpts = nrpts, inpshape=inpshape, outshape=outshape)
    train_summary, inference_summary, other_summary = {},{},{}

    for epochs in epoch_trials:
        train_pars = default_pars()
        train_pars['epochs'] = epochs
        train_pars['batch'] = 4
        model, _, duration = train_model(trainpars=train_pars, data=data)
        train_summary[epochs] = duration

    test_data['epoch_times'] = train_summary

    for nrsamples in inference_trials:
        duration = apply_model(model=model, data=data, nrsamples=nrsamples)
        inference_summary[nrsamples] = duration

    test_data['inference_times'] = inference_summary

    #0ther summary
    other_summary['Platform'] = 'Keras'
    other_summary['Input Shape'] = inpshape
    other_summary['Output Shape'] = outshape
    other_summary['Model Parameters'] = model.count_params()

    test_data['other_summary'] = other_summary

    request.node.test_data = test_data

def test_3d_seismic_imgto_img(request, test_data):
    """Test for 3D image-to-image model training."""
    inpshape = [1, 16, 16, 16]
    outshape = [1, 16, 16, 16]
    epoch_trials = list(range(5, 21, 5))
    inference_trials = [1, 8, 64, 128]
    nrpts = 32

    data = get_3d_seismic_imgtoimg_data(nrpts = nrpts, inpshape=inpshape, outshape=outshape)
    train_summary, inference_summary, other_summary = {},{},{}

    for epochs in epoch_trials:
        train_pars = default_pars()
        train_pars['epochs'] = epochs
        train_pars['batch'] = 4
        model, _, duration = train_model(trainpars=train_pars, data=data)
        train_summary[epochs] = duration

    test_data['epoch_times'] = train_summary

    for nrsamples in inference_trials:
        duration = apply_model(model=model, data=data, nrsamples=nrsamples)
        inference_summary[nrsamples] = duration

    test_data['inference_times'] = inference_summary

    #0ther summary
    other_summary['Platform'] = 'Keras'
    other_summary['Input Shape'] = inpshape
    other_summary['Output Shape'] = outshape
    other_summary['Model Parameters'] = model.count_params()

    test_data['other_summary'] = other_summary

    request.node.test_data = test_data
    