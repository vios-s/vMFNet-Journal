import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from sklearn.cluster import KMeans
from models.encoder import *
from models.decoder import *
from models.segmentor import *
from composition.model import *
from composition.helpers import *

class CompCSD(nn.Module):
    def __init__(self, image_channels, layer, num_classes, z_length, anatomy_out_channels, vMF_kappa):
        super(CompCSD, self).__init__()

        self.image_channels = image_channels
        self.layer = layer
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.num_classes = num_classes

        self.activation_layer = ActivationLayer(vMF_kappa)

        self.encoder = Encoder(self.image_channels)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes, self.layer)
        self.decoder = Decoder(self.image_channels, self.layer)

    def forward(self, x, layer=7):
        kernels = self.conv1o1.weight
        features = self.encoder(x)
        vc_activations = self.conv1o1(features[layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        content, kernel_labels = self.calculate_content(vmf_activations)
        decoding_features = self.compose(content, kernel_labels, vmf_activations)
        rec = self.decoder(decoding_features)
        pre_seg = self.segmentor(content, features)
        return rec, pre_seg, content, features[layer], kernels

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
        # self.encoder.load_state_dict(torch.load(dir_checkpoint, map_location=self.device))

    def load_vmf_kernels(self, dict_dir):
        weights = getVmfKernels(dict_dir, self.device)
        self.conv1o1 = Conv1o1Layer(weights, self.device)

    def calculate_content(self, vmf_activations):
        kernels = self.conv1o1.weight
        ##### cluster to kernels to 8 centers and get the assignment
        kernels = kernels.squeeze(2).squeeze(2).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.anatomy_out_channels, random_state=0).fit(kernels)
        kernel_labels = kmeans.labels_  # 512 list


        # vmf_activations # 4, 512, 72, 72
        # kernels # 512, 128, 1, 1]
        ###### rounding vmf_activations to content
        vmf_activations = F.softmax(vmf_activations, dim=1)
        content = torch.zeros([vmf_activations.size(0), self.anatomy_out_channels, vmf_activations.size(2), vmf_activations.size(2)])
        content = content.to(self.device)
        for k in range(vmf_activations.size(0)):
            single_vmf_activations = vmf_activations[k]
            for i in range(vmf_activations.size(1)):
                content[k, int(kernel_labels[i]), :, :] += single_vmf_activations[i, :, :]
        content = content.to(self.device)
        return content, kernel_labels

        ######################################## only assign the maximally activated kernels
        # content = torch.zeros([vmf_activations.size(0), self.anatomy_out_channels, vmf_activations.size(2), vmf_activations.size(2)])
        # for k in range(vmf_activations.size(0)):
        #     single_vmf_activations = vmf_activations[k]
        #     max_vmf_activations, max_vmf_activations_index = torch.max(single_vmf_activations, dim=0)
        #     # max_vmf_activations = max_vmf_activations.expand(single_vmf_activations.size())
        #     # norm_vmf_activations = single_vmf_activations / max_vmf_activations
        #     # norm_vmf_activations = (norm_vmf_activations > 0.5)*1.0
        #     max_vmf_activations_index = max_vmf_activations_index.detach().cpu().numpy()
        #     for i in range(vmf_activations.size(2)):
        #         for j in range(vmf_activations.size(3)):
        #             content[k, int(kernel_labels[max_vmf_activations_index[i, j]]), i, j] = 1
        #
        #
        #     # 2ed activations
        #     for i in range(vmf_activations.size(2)):
        #         for j in range(vmf_activations.size(3)):
        #             single_vmf_activations[int(max_vmf_activations_index[i, j]), i, j] = 0
        #     max_vmf_activations, max_vmf_activations_index = torch.max(single_vmf_activations, dim=0)
        #     max_vmf_activations_index = max_vmf_activations_index.detach().cpu().numpy()
        #     for i in range(vmf_activations.size(2)):
        #         for j in range(vmf_activations.size(3)):
        #             content[k, int(kernel_labels[max_vmf_activations_index[i, j]]), i, j] = 1
        # content = content.to(self.device)

        ############################################# channel wise clustering
        # channel_cluster_centers = []
        # for k in range(vmf_activations.size(0)):
        #     cluster_activations = vmf_activations[k].reshape((512, 72 * 72)).detach().cpu().numpy()
        #     kmeans = KMeans(n_clusters=self.anatomy_out_channels, random_state=0).fit(cluster_activations)
        #     channel_cluster_centers.append(kmeans.cluster_centers_.reshape(8, 72, 72))
        # content = np.concatenate((channel_cluster_centers[0].reshape(1, 8, 72, 72),
        #                           channel_cluster_centers[1].reshape(1, 8, 72, 72),
        #                           channel_cluster_centers[2].reshape(1, 8, 72, 72),
        #                           channel_cluster_centers[3].reshape(1, 8, 72, 72)), axis=0)
        # content = torch.from_numpy(content).to(self.device)

        # ############################################# channel wise clustering accumulation
        # content = np.zeros(
        #     [vmf_activations.size(0), self.anatomy_out_channels, vmf_activations.size(2), vmf_activations.size(3)])
        # for k in range(vmf_activations.size(0)):
        #     cluster_activations = vmf_activations[k].reshape((512, 72 * 72)).detach().cpu().numpy()
        #     kmeans = KMeans(n_clusters=self.anatomy_out_channels, random_state=0).fit(cluster_activations)
        #     cluster_labels = kmeans.labels_
        #     for h in range(512):
        #         content[k, int(cluster_labels[h]), :, :] += vmf_activations[k, h, :, :].detach().cpu().numpy()
        #
        # content = torch.from_numpy(content).to(self.device)

    def compose(self, content, kernel_labels, vmf_activations):
        # load the kernels
        kernels = self.conv1o1.weight #[512, 128, 1, 1]
        kernels = kernels.squeeze(2).squeeze(2) # [512, 128]

        # [4, 64, 144, 144]
        features = torch.zeros([vmf_activations.size(0), kernels.size(1), vmf_activations.size(2), vmf_activations.size(3)])
        features = features.to(self.device)

        vmf_activations = F.softmax(vmf_activations, dim=1)
        # normalise vmf_activations [4, 512, 72, 72] here:
        for k in range(vmf_activations.size(0)):
            single_content = content[k]
            single_vmf_activations = vmf_activations[k].detach()
            # max_vmf_activations, max_vmf_activations_index = torch.max(single_vmf_activations, dim=0)
            # max_vmf_activations = max_vmf_activations.expand(single_vmf_activations.size())
            # norm_vmf_activations = single_vmf_activations / max_vmf_activations.detach() #[512, 72, 72]

            for i in range(single_content.size(0)):
                single_vmf_activations[[j for j in range(len(kernel_labels)) if kernel_labels[j] == i], :, :] *= single_content[i]

            norm_vmf_activations = torch.permute(single_vmf_activations, (1, 2, 0))  # [512, 72, 72]
            # decompose the features for each kernelx
            feature = torch.matmul(norm_vmf_activations, kernels.detach())
            feature = torch.permute(feature, (2, 0, 1))
            features[k, :, :, :] = feature.to(self.device)

        return features

