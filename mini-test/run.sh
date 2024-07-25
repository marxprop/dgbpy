faultidmodeldir = /opt/apps/OpendTect/oxy_MLModels/oxy_fault_id.h5
opendtectPythonDir = /opt/apps/OpendTect/Python

# Set python environment in system path and activate the environment
export PATH=$opendtectPythonDir/bin:$PATH

# Activate the environment
source $opendtectPythonDir/bin/activate 

# Install the required packages
pip install pytest
pip install tabulate

# Go to the test directory
cd mini-test

# Run the test
pytest --modelpath $faultidmodeldir --shape 96