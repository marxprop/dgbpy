#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# AUTHOR   : A. Huck
# DATE     : June 2019
#
# Deep learning apply server
#
#

import argparse
import numpy as np
from os import path
import selectors
import socket
import sys
import time
import traceback

from odpy.common import *
from odpy import oscommand
import dgbpy.keystr as dgbkeys
import dgbpy.hdf5 as dgbhdf5
import dgbpy.deeplearning_apply_clientlib as applylib

sel = selectors.DefaultSelector()

# -- command line parser

parser = argparse.ArgumentParser(
          description='Client application of a trained machine learning model')
parser.add_argument( '-v', '--version',
            action='version',version='%(prog)s 2.0')
datagrp = parser.add_argument_group( 'Data' )
datagrp.add_argument( 'modelfile', type=argparse.FileType('r'),
                       help='The input trained model file' )
datagrp.add_argument( '--examplefile', 
                       dest='examples', metavar='file', nargs='?',
                       type=argparse.FileType('r'),
                       help='Examples file to be applied' )
netgrp = parser.add_argument_group( 'Network' )
netgrp.add_argument( '--address',
            dest='addr', metavar='ADDRESS', action='store',
            type=str, default='localhost',
            help='Address to listen on' )
netgrp.add_argument( '--port',
            dest='port', action='store',
            type=int, default=65432,
            help='Port to listen on')
loggrp = parser.add_argument_group( 'Logging' )
loggrp.add_argument( '--log',
            dest='logfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Progress report output' )
loggrp.add_argument( '--syslog',
            dest='sysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='System log' )
loggrp.add_argument( '--server-log',
            dest='servlogfile', metavar='file', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Python server log' )
loggrp.add_argument( '--server-syslog',
            dest='servsysout', metavar='stdout', nargs='?',
            type=argparse.FileType('w'), default=sys.stdout,
            help='Python server System log' )
# optional
parser.add_argument( '--fakeapply', dest='fakeapply', action='store_true',
                     default=False,
                     help="applies a numpy average instead of the model" )


args = vars(parser.parse_args())
initLogging( args )
modelfnm = args['modelfile'].name

servscriptfp =  path.join(path.dirname(__file__),'deeplearning_apply-server.py')
servercmd = list()
servercmd.append( oscommand.getPythonExecNm() )
servercmd.append( servscriptfp )
servercmd.append( modelfnm )
servercmd.append( '--address' )
servercmd.append( str(args['addr']) )
servercmd.append( '--port' )
servercmd.append( str(args['port']) )
servercmd.append( '--log' )
servercmd.append( args['servlogfile'].name )
servercmd.append( '--syslog' )
servercmd.append( args['servsysout'].name )
if args['fakeapply']:
  servercmd.append( '--fakeapply' )

serverproc = oscommand.execCommand( servercmd, True )
time.sleep( 2 )

def getApplyTrace( dict ):
  arr3d = dict['arr']
  stepout = dict['stepout']
  idx = dict['idx']
  idy = dict['idy']
  return arr3d[:,idx-stepout[0]:idx+stepout[0]+1,\
                 idy-stepout[1]:idy+stepout[1]+1,:]

def create_request(action, value=None):
  if value == None:
    return dict(
      type="text/json",
      encoding="utf-8",
      content=dict(action=action),
    )
  elif action == 'outputs':
    return dict(
      type="text/json",
      encoding="utf-8",
      content=dict(
        action=action,
        value= {
          'names': value,
          dgbkeys.dtypepred: 'uint8',
          dgbkeys.dtypeprob: 'float32',
          dgbkeys.dtypeconf: 'float32'
        },
      ),
    )
  elif action == 'data':
    arr = getApplyTrace(value)
    return dict(
      type='binary/array',
      encoding=[arr.dtype.name],
      content=[arr],
    )
  else:
    return dict(
        type="binary/custom-client-binary-type",
        encoding="binary",
        content=bytes(action + value, encoding="utf-8"),
    )

def req_connection(host, port, request):
  addr = (host, port)
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setblocking(True)
  sock.connect_ex(addr)
  events = selectors.EVENT_READ | selectors.EVENT_WRITE
  message = applylib.Message(sel, sock, addr, request)
  sel.register(sock, events, data=message)

def getApplyPars( args ):
  if args['examples'] == None:
    ret= {
      'stepout': [16,16,16],
      'nrattribs': 1,
      'outputnms': dgbhdf5.getOutputNames(modelfnm,[0])
    }
  else:
    exfnm = args['examples'].name
    info = dgbhdf5.getInfo( exfnm )
    stepout = info['stepout']
    if not isinstance(stepout,list):
      stepout=[0,0,stepout]
    ret = {
      'stepout': stepout,
      'nrattribs': dgbhdf5.get_nr_attribs( info ),
      'outputnms': dgbhdf5.getOutputs( exfnm )
    }
  return ret

pars = getApplyPars( args )
stepout = pars['stepout']

nrattribs = pars['nrattribs']
nrlines_out = 1
nrtrcs_out = 100
nrlines_in = nrlines_out + 2 * stepout[0]
nrtrcs_in = nrtrcs_out + 2 * stepout[1]
nrz_in = 463
nrz_out = nrz_in - 2 * stepout[2]
inpdata = np.random.random( nrattribs*nrlines_in*nrtrcs_in*nrz_in )
inpdata = inpdata.reshape((nrattribs,nrlines_in,nrtrcs_in,nrz_in))
inpdata = inpdata.astype('float32')

start = time.time()

host,port = args['addr'], args['port']
req_connection(host, port, create_request('status'))
req_connection(host, port, create_request('outputs',pars['outputnms']))
applydict = {
  'arr': inpdata,
  'stepout': stepout,
  'idx': stepout[0],
  'idy': stepout[1]
}
lastidy = nrtrcs_in-stepout[1]
nrrepeats = 1
for i in range(nrrepeats):
  for idy in range(stepout[1],nrtrcs_in-stepout[1]):
    applydict['idy'] = idy
    req_connection(host, port, create_request('data',applydict))

req_connection(host, port, create_request('kill'))

try:
  while True:
    events = sel.select(timeout=1)
    for key, mask in events:
      message = key.data
      try:
        message.process_events(mask)
      except Exception:
        std_msg(
            "main: error: exception for",
            f"{message.addr}:\n{traceback.format_exc()}",
        )
        message.close()
    # Check for a socket being monitored to continue.
    if not sel.get_map():
      break
except KeyboardInterrupt:
  std_msg("caught keyboard interrupt, exiting")
finally:
  sel.close()
  oscommand.kill( serverproc )
  duration = time.time()-start
  log_msg( "Total time:",  "{:.3f}".format(duration), "s.;", \
         "{:.3f}".format(nrtrcs_out/duration), "tr/s." )