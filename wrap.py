import ctypes
import os
from ctypes.util import find_library
import numpy as np
import numpy.ctypeslib as npct
import PIL.Image
# find and load the library

# OSX or linux

basepath = os.path.dirname(os.path.abspath(__file__))
dllspath = os.path.join(basepath, 'libdarknet.so')
os.environ['PATH'] = dllspath + os.pathsep + os.environ['PATH']

libdarknet = ctypes.cdll.LoadLibrary(os.path.join(basepath, 'libdarknet.so'))


c_float_p = ctypes.POINTER(ctypes.c_float)

array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

# set the argument type
# (char *datacfg, char *cfgfile, char *weightfile, float* data, int h, int w, int c, float thresh, float hier_thresh)
libdarknet.test_detector_python.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                            ctypes.c_char_p, array_1d_float,
                                            ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_float, ctypes.c_float, array_1d_float]

# set the return type
#libdarknet.test_detector_python.restype = c_float_p


def detector(datacfg, cfgfile, weightfile, data, h, w, c, thresh, hier_thresh):
    ''' a detector '''
    datacfg_c = ctypes.c_char_p(datacfg)
    cfgfile_c = ctypes.c_char_p(cfgfile)
    weightfile_c = ctypes.c_char_p(weightfile)

    h_c = ctypes.c_int(h)
    w_c = ctypes.c_int(w)
    c_c = ctypes.c_int(c)
    thresh_c = ctypes.c_float(thresh)
    hier_thresh_c = ctypes.c_float(hier_thresh)

    data = data.astype(np.float32)
    data_p = data.ctypes.data_as(c_float_p)

    out = np.empty(13 * 13 * 115 * (4 + 18), dtype="float32")
    out_p = out.ctypes.data_as(c_float_p)
    libdarknet.test_detector_python(datacfg_c, cfgfile_c, weightfile_c,
                                    data.flatten(), h_c, w_c, c_c, thresh_c,
                                    hier_thresh_c, out)
    boxes = out[:13*13*115*4].reshape((-1, 4))
    confs = out[13*13*115*4:].reshape((-1, 18))
    return boxes, confs
im = PIL.Image.open(
    "/home/pierre/lego-assembly-helper/multiblocks_val/images/0b2b390a-c47d-"
    "4a4f-b9bf-45c2564e1f16.jpg")
import pdb
pdb.set_trace()
boxes, confs = detector("/home/pierre/darknet/cfg/voc.data",
                        "/home/pierre/darknet/cfg/yolo-voc.cfg",
                        "/home/pierre/backup/5blocksyolo-voc_9000.weights",
                        np.array(im), im.size[0], im.size[1], 3, 0.24, 0.5)
