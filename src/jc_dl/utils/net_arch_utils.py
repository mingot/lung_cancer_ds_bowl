import os
import sys

import numpy as np

from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve

models_path = os.environ['LUNG_PATH'] + "/models/"
weights_path = os.environ['LUNG_PATH'] + "/models/weights/"

def download_file(fname, origin):
    fpath = os.path.join(models_path, fname)
    if os.path.exists(fpath):
        return fpath


    print('Downloading weights from: ',  origin)
    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath)
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(fpath):
            os.remove(fpath)
        raise e

    return fpath

def load_weights(net, weights_filename):
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    net.load_weights(os.path.join(weights_path, weights_filename))

def save_weights(net, weights_filename):
    net.save_weights(os.path.join(weights_path, weights_filename))