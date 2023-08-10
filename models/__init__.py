from .compcsd import *
from .unet_model import *
from .unet_parts import *
from .weight_init import *
from .blocks import *
from .compcsd2 import *
from .compcsd2weak import *
from .discriminator import *
from .classifier import *

import sys

def get_dis(params):
    return Discriminator(params['image_channels'])

def get_model(name, params):
    if name == 'compcsd':
        return CompCSD(params['image_channels'], params['layer'], params['num_classes'], params['z_length'], params['anatomy_out_channels'], params['vMF_kappa'])
    if name == 'compcsd2':
        return CompCSD2(params['image_channels'], params['layer'], params['vc_numbers'], params['num_classes'], params['z_length'],
                       params['anatomy_out_channels'], params['vMF_kappa'])
    if name == 'compcsd2weak':
        return CompCSD2weak(params['image_channels'], params['layer'], params['vc_numbers'], params['num_classes'], params['z_length'],
                       params['anatomy_out_channels'], params['vMF_kappa'])
    if name == 'unet':
        return UNet(params['num_classes'])
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)