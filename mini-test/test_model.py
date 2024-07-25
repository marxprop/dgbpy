import pytest

import sys
sys.path.insert(0, '..')

import time
import dgbpy.mlapply as dgbml
import dgbpy.keystr as dbk
import dgbpy.mlio as dgbmlio   
import dgbpy.zipmodelbase as zm
import gc

from init_data import *

@pytest.fixture
def test_data(request):
    """Fixture to store custom test data."""
    return {}

def count_parameters(model):
    if isinstance(model, zm.ZipPredictModel):
        model = model.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_faultid_model(request, test_data, filepath, shape):
    inference_trials = [1, 8, 16, 64, 128]
    inference_summary, other_summary = {},{}

    if not filepath:
        pytest.skip("Skipped: Model path not provided"),
    
    batchsize = 4
    input_shape = [shape, shape, shape]
    output_shape = [shape, shape, shape]


    for nrsamples in inference_trials:
        try:
            start_time = time.time()
            # Generate sample test data
            data = get_3d_seismic_imgtoimg_data(nrpts=nrsamples, split = 0, inpshape=input_shape, outshape=output_shape)
            samples = data[dbk.xtraindictstr]

            # Load the model
            model,info = dgbmlio.getModel(filepath)

            #Apply the model
            applyInfo = dgbmlio.getApplyInfo( info )
            dgbml.doApply(model, info, samples, None, applyInfo, batchsize)
            end_time = time.time()
            duration = end_time - start_time
            inference_summary[samples.shape] = duration

            gc.collect()
        except Exception as e:
            continue

    test_data['inference_times'] = inference_summary

    # Count model parameters
    total_params, trainable_params = count_parameters(model)

     #0ther summary
    other_summary['Platform'] = 'Pytorch'
    other_summary['Input Shape'] = input_shape
    other_summary['Output Shape'] = output_shape
    other_summary['Total Model Parameters'] = total_params
    other_summary['Trainable Model Parameters'] = trainable_params

    test_data['other_summary'] = other_summary

    request.node.test_data = test_data

    
    


            