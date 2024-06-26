import sys
sys.path.insert(0, '..')
import dgbpy.keystr as dbk
import dgbpy.dgbscikit as dgbscikit
import dgbpy.hdf5 as dgbhdf5
import dgbpy.mlapply as dgbml
import dgbpy.mlio as dgbmlio
import numpy as np


def get_default_examples(nr_inattr=1, nr_outattr=1):
    dummy_collection = get_n_collection(nr_inattr)
    targets = [f"Dummy{colnm}" for colnm in range(1, nr_outattr+1)]
    retinfo = {
        "Dummy": {
            "target": targets,
            "id": 0,
            dbk.collectdictstr: dummy_collection
        }
    }
    return retinfo

def get_n_collection(count = 1):
    retinfo = {}
    for n in range(count):
        retinfo[f"Dummy{n}"] = {"id": f"{n}"}
    return retinfo

def get_default_input(nr_inattr=1):
    if nr_inattr < 1: nr_inattr = 1

    dummy_collection = get_n_collection(nr_inattr)
    
    retinfo = {
        "Dummy": {
            dbk.collectdictstr: dummy_collection,
            "id": 0,
            "scale": dgbscikit.StandardScaler(),
        }
    }
    return retinfo

def get_default_multiple_input():
    retinfo = {
        "Dummy": {
            dbk.collectdictstr: {
                "Dummy":  {"id": 0},
                "Dummy2": {"id": 1},
                "Dummy3": {"id": 2},
            },
            "id": 0,
            "scale": dgbscikit.StandardScaler(),
        }
    }
    return retinfo


def get_dataset_dict(nrpts):
    retinfo = {"Dummy": {"Dummy": list(range(nrpts))}}
    return retinfo

def get_dataset_dict_multiple(nrpts):
    retinfo = {
        "Dummy": {
            "Dummy": list(range(nrpts)),
            "Dummy2": list(range(nrpts)),
            "Dummy3": list(range(nrpts)),
        }
    }
    return retinfo

def getCurrentConditions(conditions, step):
    current_conditions = {}
    for key, value in conditions.items():
        current_conditions[key] = value[step]
    return current_conditions

def get_default_info(nr_inattr=1, nroutattr=1):
    retinfo = {
        dbk.learntypedictstr: dbk.loglogtypestr,
        dbk.segmentdictstr: False,
        dbk.inpshapedictstr: 1,
        dbk.outshapedictstr: 1,
        dbk.classdictstr: False,
        dbk.interpoldictstr: False,
        dbk.exampledictstr: get_default_examples(nr_inattr, nroutattr),
        dbk.inputdictstr: get_default_input(nr_inattr),
        dbk.filedictstr: "dummy",
        dbk.estimatedsizedictstr: 1,
    }
    return retinfo

def getNrDims(inpshape):
    if isinstance(inpshape, int):
        if inpshape > 1:
            return 1
        else:
            return inpshape
    else:
        return len(inpshape) - inpshape.count(1)

def getExampleInfos(infodict):
    learntype = infodict[dbk.learntypedictstr]
    classification = infodict[dbk.classdictstr]
    inpshape = infodict[dbk.inpshapedictstr]
    nrattribs = dgbhdf5.getNrAttribs(infodict)
    nrdims = getNrDims(inpshape)
    return (learntype, classification, nrdims, nrattribs)


def prepare_dataset_dict(info, nbchunks=1, seed=0, split=0.2, nbfolds=0):
    dsets = dgbmlio.getChunks(info[dbk.datasetdictstr], nbchunks)
    datasets = []
    for dset in dsets:
        if dgbhdf5.isLogInput(info) and nbfolds:
            datasets.append(
                dgbmlio.getCrossValidationIndices(
                    dset, seed=seed, valid_inputs=split, nbfolds=nbfolds
                )
            )
        else:
            datasets.append(dgbmlio.getDatasetNms(dset, validation_split=split))
    info.update({dbk.trainseldicstr: datasets, dbk.seeddictstr: seed})
    return info

def prepare_array_by_info(infos, ifold, ichunk):
    if ifold and dgbhdf5.isCrossValidation( infos ):
        datasets = infos[dbk.trainseldicstr][ichunk][dbk.foldstr+f'{ifold}']
    else:
        datasets = infos[dbk.trainseldicstr][ichunk]


def prepare_data_arr(info, split, nrpts):
    valid_nrpts = int(split * nrpts)
    inp_shape = info[dbk.inpshapedictstr]
    out_shape = info[dbk.outshapedictstr]

    inpnrattribs = dgbhdf5.getNrAttribs(info)
    outnrattribs = dgbhdf5.getNrOutputs(info)
    x_train_shape = dgbhdf5.get_np_shape(inp_shape, nrpts, inpnrattribs)
    if isinstance(out_shape, int):
        out_shape = (out_shape,)
        if 1 not in out_shape:
            out_shape = (1, *out_shape)
    else:
        out_shape = (outnrattribs, *out_shape)
    y_train_shape = (nrpts, *out_shape)

    x_valid_shape = dgbhdf5.get_np_shape(inp_shape, valid_nrpts, inpnrattribs)
    y_valid_shape = (valid_nrpts, *out_shape)

    if dgbhdf5.isClassification(info):
        nclasses = dgbhdf5.getNrClasses(info)
    else:
        nclasses = None
    return nclasses, (x_train_shape, y_train_shape), (x_valid_shape, y_valid_shape)

def get_seismic_imgtoimg_info(nrclasses=5, nr_inattr=1, nr_outattr=1, inpshape=[1,8,8], outshape=[1,8,8]):
    if len(inpshape) != 3 or len(outshape) != 3:
        raise ValueError("Input shape and output shape must be 3D")
    
    default = get_default_info(nr_inattr, nr_outattr)
    default[dbk.learntypedictstr] = dbk.seisimgtoimgtypestr
    default[dbk.inpshapedictstr] = inpshape
    default[dbk.outshapedictstr] = outshape
    default[dbk.classdictstr] = nrclasses > 1
    default[dbk.interpoldictstr] = True
    default[dbk.classesdictstr] = list(range(1, nrclasses+1))
    return default


def get_seismic_classification_info(nrclasses=5, nr_inattr=1, nr_outattr=1, ):
    default = get_default_info(nr_inattr, nr_outattr)
    default[dbk.learntypedictstr] = dbk.seisclasstypestr
    default[dbk.inpshapedictstr] = [1, 8, 8]
    default[dbk.outshapedictstr] = 1
    default[dbk.classdictstr] = True
    default[dbk.interpoldictstr] = True
    default[dbk.classesdictstr] = list(range(1, nrclasses+1))
    return default


def get_loglog_info():
    default = get_default_info()
    default[dbk.exampledictstr] = get_default_examples(nr_inattr=3, nr_outattr=1)
    default[dbk.inputdictstr] = get_default_multiple_input()
    default[dbk.learntypedictstr] = dbk.loglogtypestr
    default[dbk.inpshapedictstr] = 1
    default[dbk.outshapedictstr] = 1
    default[dbk.classdictstr] = False
    default[dbk.interpoldictstr] = False
    return default

def loglog_classification_info(nrclasses=3):
    default = get_default_info()
    default[dbk.exampledictstr] = get_default_examples(nr_inattr=3, nr_outattr=1)
    default[dbk.inputdictstr] = get_default_multiple_input()
    default[dbk.learntypedictstr] = dbk.loglogtypestr
    default[dbk.inpshapedictstr] = 1
    default[dbk.outshapedictstr] = 1
    default[dbk.classdictstr] = nrclasses > 1
    default[dbk.interpoldictstr] = False
    default[dbk.classesdictstr] = list(range(1, nrclasses+1))
    return default

def get_2d_seismic_imgtoimg_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0, nrclasses=5, nr_inattr=1, nr_outattr=1, inpshape=[1,8,8], outshape=[1,8,8]):
    info = get_seismic_imgtoimg_info(nrclasses, nr_inattr, nr_outattr, inpshape, outshape)
    dataset = get_dataset_dict(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)

    if nrclasses > 1:
        y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
        y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)
    else:
        y_train = np.random.random(y_train_shape).astype(np.single)
        y_validate = np.random.random(y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }

def get_3d_seismic_imgtoimg_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0, nrclasses=5, nr_inattr=1, nr_outattr=1, inpshape = [1,8,8], outshape=[1,8,8]):
    info = get_seismic_imgtoimg_info(nrclasses, nr_inattr, nr_outattr, inpshape, outshape)
    info[dbk.inpshapedictstr] = [8, 8, 8]
    info[dbk.outshapedictstr] = [8, 8, 8]
    dataset = get_dataset_dict(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)

    if nrclasses > 1:
        y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
        y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)
    else:
        y_train = np.random.random(y_train_shape).astype(np.single)
        y_validate = np.random.random(y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }


def get_seismic_classification_data(nrpts=40, nbchunks=1, seed=0, split=0.2, nbfolds=0, nrclasses=5, nr_inattr=1, nr_outattr=1):
    info = get_seismic_classification_info(nrclasses, nr_inattr, nr_outattr)
    dataset = get_dataset_dict(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }


def get_loglog_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0, flatten=False):
    info = get_loglog_info()
    dataset = get_dataset_dict_multiple(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    _, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.random(y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.random(y_valid_shape).astype(np.single)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_validate = x_validate.reshape(x_validate.shape[0], -1)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }


def get_loglog_classification_data(nrpts=16, nbchunks=1, seed=0, split=0.2, nbfolds=0, flatten=False):
    info = loglog_classification_info()
    dataset = get_dataset_dict_multiple(nrpts)
    info[dbk.datasetdictstr] = dataset
    info = prepare_dataset_dict(info, nbchunks, seed, split, nbfolds)

    nclasses, train_shape, valid_shape = prepare_data_arr(info, split, nrpts)
    x_train_shape, y_train_shape = train_shape
    x_valid_shape, y_valid_shape = valid_shape

    x_train = np.random.random(x_train_shape).astype(np.single)
    y_train = np.random.randint(nclasses, size=y_train_shape).astype(np.single)
    x_validate = np.random.random(x_valid_shape).astype(np.single)
    y_validate = np.random.randint(nclasses, size=y_valid_shape).astype(np.single)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_validate = x_validate.reshape(x_validate.shape[0], -1)

    return {
        dbk.xtraindictstr: x_train,
        dbk.ytraindictstr: y_train,
        dbk.xvaliddictstr: x_validate,
        dbk.yvaliddictstr: y_validate,
        dbk.infodictstr: info,
    }
