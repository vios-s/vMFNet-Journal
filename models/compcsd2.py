import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode
import cv2
import random
from models.encoder import *
from models.decoder import *
from models.segmentor import *
from composition.model import *
from composition.helpers import *
from models.content_iteration import *

class CompCSD2(nn.Module):
    def __init__(self, image_channels, layer, vc_numbers, num_classes, z_length, anatomy_out_channels, vMF_kappa):
        super(CompCSD2, self).__init__()

        self.image_channels = image_channels
        self.layer = layer
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.num_classes = num_classes
        self.vc_num = vc_numbers

        self.activation_layer = ActivationLayer(vMF_kappa)
        self.content_iterater = ContentIter(self.anatomy_out_channels, self.vc_num)

        self.encoder = Encoder(self.image_channels)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.vc_num, self.num_classes, self.layer)
        self.decoder = Decoder(self.image_channels, self.anatomy_out_channels, self.layer)

    def forward(self, x, layer=7):
        kernels = self.conv1o1.weight
        features = self.encoder(x)
        vc_activations = self.conv1o1(features[layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        # norm_vmf_activations = F.softmax(vmf_activations, dim=1)
        self.vmf_activations = norm_vmf_activations
        self.vc_activations = vc_activations
        content = self.calculate_content(norm_vmf_activations)
        decoding_features = self.compose(content, vc_activations, norm_vmf_activations)
        rec = self.decoder(decoding_features)
        # pre_seg = TVF.resize(content, ((2**(9-layer))*content.size(2), (2**(9-layer))*content.size(3)), interpolation=InterpolationMode.NEAREST)
        pre_seg = self.segmentor(norm_vmf_activations, features)

        # new_vmf_activations = self.manipulate_content(content, norm_vmf_activations, vc_activations)
        # # new_vmf_activations = F.softmax(new_vmf_activations, dim=1)
        # new_decoding_features = self.compose(content, vc_activations, new_vmf_activations)
        # new_rec = self.decoder(new_decoding_features)
        #
        # reg_features = self.encoder(new_rec)
        # reg_vc_activations = self.conv1o1(reg_features[layer]) # fp*uk
        # reg_vmf_activations = self.activation_layer(reg_vc_activations) # L
        # reg_norm_vmf_activations = torch.zeros_like(reg_vmf_activations)
        # reg_norm_vmf_activations = reg_norm_vmf_activations.to(self.device)
        # for i in range(reg_vmf_activations.size(0)):
        #     reg_norm_vmf_activations[i, :, :, :] = F.normalize(reg_vmf_activations[i, :, :, :], p=1, dim=0)
        # reg_norm_vmf_activations = F.softmax(reg_vmf_activations, dim=1)
        # reg_content = self.calculate_content(reg_vmf_activations)

        return rec, pre_seg, content, features[layer], kernels, norm_vmf_activations

    def load_encoder_weights(self, dir_checkpoint, device):
        self.device = device
        pre_trained = torch.load(dir_checkpoint, map_location=self.device) # without unet.
        new = list(pre_trained.items())

        my_model_kvpair = self.encoder.state_dict() # with unet.
        count = 0
        for key in my_model_kvpair:
            layer_name, weights = new[count]
            my_model_kvpair[key] = weights
            count += 1
        self.encoder.load_state_dict(my_model_kvpair)

    def load_vmf_kernels(self, dict_dir):
        weights = getVmfKernels(dict_dir, self.device)
        self.conv1o1 = Conv1o1Layer(weights, self.device)

    def calculate_content(self, vmf_activations):
        condition = torch.zeros(vmf_activations.size(0), 1, vmf_activations.size(2), vmf_activations.size(3))
        condition = condition.to(self.device)
        content = self.content_iterater(vmf_activations, condition)
        content = torch.sigmoid(content)
        return content

    def compose(self, content, vc_activations, vmf_activations, new_content=None):
        kernels = self.conv1o1.weight #[512, 128, 1, 1]
        kernels = kernels.squeeze(2).squeeze(2) # [512, 128]

        features = torch.zeros([vmf_activations.size(0), kernels.size(1), vmf_activations.size(2), vmf_activations.size(3)])
        features = features.to(self.device)

        for k in range(vmf_activations.size(0)):
            # single_content = content[k].detach()
            single_vmf_activations = vmf_activations[k]
            # single_vc_activations = vc_activations[k]

            # single_vc_activations = torch.permute(single_vc_activations, (1, 2, 0))  # [512, 72, 72]
            single_vmf_activations = torch.permute(single_vmf_activations, (1, 2, 0))  # [512, 72, 72]
            feature = torch.matmul(single_vmf_activations, kernels)
            feature = torch.permute(feature, (2, 0, 1))
            features[k, :, :, :] = feature
        return features


    def manipulate_content(self, content, norm_vmf_activations, vc_activations):
        with torch.no_grad():
            old_content = content.detach()
            old_vmf_activations = norm_vmf_activations.detach()
            old_vc_activations = vc_activations.detach()

            new_content = torch.zeros_like(old_vmf_activations)
            new_content = new_content.to(self.device)

            random_index = random.randint(0, self.vc_num-1)

            for i in range(self.vc_num):
                if i == random_index:
                    pass
                else:
                    new_content[:, i, :, :] = old_vmf_activations[:, i, :, :]
        return new_content

