import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import sys
import time
from torchvision.transforms import InterpolationMode
from sklearn.cluster import KMeans
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
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes, self.layer)
        self.decoder = Decoder(self.image_channels, self.anatomy_out_channels, self.layer)

    def forward(self, x, layer=7):
        kernels = self.conv1o1.weight
        features = self.encoder(x)
        vc_activations = self.conv1o1(features[layer]) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        content = self.calculate_content(vmf_activations)
        decoding_features = self.compose(content, vc_activations, vmf_activations)
        rec = self.decoder(decoding_features)

        # new_rec, new_lv = self.manipulate_content(content, vmf_activations, vc_activations)
        # new_rec, lv_weight = self.manipulate_style(content, vmf_activations, vc_activations)
        # pre_seg = self.segmentor(content, features)
        pre_seg = TVF.resize(content, ((10-layer)*content.size(2), (10-layer)*content.size(3)), interpolation=InterpolationMode.NEAREST)
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
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(self.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        condition = torch.zeros(vmf_activations.size(0), 1, vmf_activations.size(2), vmf_activations.size(3))
        condition = condition.to(self.device)
        content = self.content_iterater(norm_vmf_activations, condition)
        content = F.softmax(content, dim=1)
        return content

        # kernels = self.conv1o1.weight
        # ##### cluster to kernels to 8 centers and get the assignment
        # kernels = kernels.squeeze(2).squeeze(2).detach().cpu().numpy()
        # kmeans = KMeans(n_clusters=self.anatomy_out_channels, random_state=0).fit(kernels)
        # kernel_labels = kmeans.labels_  # 512 list
        #
        #
        # # vmf_activations # 4, 512, 72, 72
        # # kernels # 512, 128, 1, 1]
        # ###### rounding vmf_activations to content
        # vmf_activations = F.softmax(vmf_activations, dim=1)
        # content = torch.zeros([vmf_activations.size(0), self.anatomy_out_channels, vmf_activations.size(2), vmf_activations.size(2)])
        # content = content.to(self.device)
        # for k in range(vmf_activations.size(0)):
        #     single_vmf_activations = vmf_activations[k]
        #     for i in range(vmf_activations.size(1)):
        #         content[k, int(kernel_labels[i]), :, :] += single_vmf_activations[i, :, :]
        # content = content.to(self.device)
        # return content, kernel_labels

    def compose_new(self, content, vc_activations, vmf_activations, new_content=None):
        kernels = self.conv1o1.weight #[512, 128, 1, 1]
        kernels = kernels.squeeze(2).squeeze(2) # [512, 128]

        features = torch.zeros([vc_activations.size(0), kernels.size(1), vc_activations.size(2), vmf_activations.size(3)])
        features = features.to(self.device)

        for k in range(vc_activations.size(0)):
            single_content = content[k].detach()
            single_vmf_activations = vmf_activations[k]
            single_vc_activations = vc_activations[k]

            single_vc_activations = torch.permute(single_vc_activations, (1, 2, 0))  # [512, 72, 72]
            # single_vmf_activations = torch.permute(single_vmf_activations, (1, 2, 0))  # [512, 72, 72]
            feature = torch.matmul(single_vc_activations, kernels.detach())
            feature = torch.permute(feature, (2, 0, 1))
            if new_content is not None:
                single_new_content = new_content[k].detach()
                attention = torch.abs((single_content>0.5)*1.0 - (single_new_content>0.5)*1.0)
                attention = torch.sum(attention, dim=0)
                attention_mask = (attention>0)*1.0
                attention_mask = 1 - attention_mask
                attention_mask = attention_mask.expand(feature.size())
                attention_mask = attention_mask.detach()
                feature = feature*attention_mask
            features[k, :, :, :] = feature
        if new_content is not None:
            features = torch.cat((new_content.detach(), features), dim=1)
        else:
            features = torch.cat((content.detach(), features), dim=1)
        return features

    def compose(self, content, vc_activations, vmf_activations):
        # load the kernels
        kernels = self.conv1o1.weight #[512, 128, 1, 1]
        kernels = kernels.squeeze(2).squeeze(2) # [512, 128]

        # # [4, 64, 144, 144]
        # features = torch.zeros([vc_activations.size(0), kernels.size(1)*self.anatomy_out_channels, vc_activations.size(2), vmf_activations.size(3)])
        # features = features.to(self.device)
        #
        # # normalise vmf_activations [4, 512, 72, 72] here:
        # for k in range(vc_activations.size(0)):
        #     single_content = content[k].detach()
        #     single_vmf_activations = vmf_activations[k].detach()
        #     # single_vmf_activations = F.normalize(single_vmf_activations, p=1, dim=0)
        #     single_vc_activations = vc_activations[k]
        #
        #     # attention = torch.zeros_like(single_vc_activations)
        #     # attention = attention.to(self.device)
        #
        #     feature = []
        #     ######## Use the content as attention to manipulate the composition
        #     ######## Better way to do the masking here
        #     single_vmf_activations = single_vmf_activations > 0
        #     # print(torch.sum(single_vmf_activations))
        #     for i in range(self.anatomy_out_channels):
        #         content_attention = single_content[i, :, :].expand(single_vmf_activations.size())
        #         attention = single_vmf_activations*content_attention
        #
        #         single_vc_activations = attention * single_vc_activations
        #         single_vc_activations_permute = torch.permute(single_vc_activations, (1, 2, 0))  # [512, 72, 72]
        #         # decompose the features for each kernelx
        #         local_feature = torch.matmul(single_vc_activations_permute, kernels.detach())
        #         local_feature = torch.permute(local_feature, (2, 0, 1))
        #         feature.append(local_feature)
        #     feature = torch.cat(feature, dim=0) # 64xself.anatomy_out_channels, 144, 144
        #     features[k, :, :, :] = feature
        #
        # return features


        features = torch.zeros([vc_activations.size(0), kernels.size(1), vc_activations.size(2), vmf_activations.size(3)])
        features = features.to(self.device)

        for k in range(vc_activations.size(0)):
            single_content = content[k].detach()
            single_vmf_activations = vmf_activations[k].detach()
            single_vc_activations = vc_activations[k]

            # attention = torch.zeros_like(single_vc_activations)
            # attention = attention.to(self.device)
            #
            # ######## Use the content as attention to manipulate the composition
            # ######## Better way to do the masking here
            # single_vmf_activations = single_vmf_activations > 0
            # for i in range(self.anatomy_out_channels):
            #     content_attention = single_content[i, :, :].expand(single_vmf_activations.size())
            #     attention += single_vmf_activations*content_attention
            #
            # single_vc_activations = attention * single_vc_activations
            single_vc_activations = torch.permute(single_vc_activations, (1, 2, 0))  # [512, 72, 72]
            feature = torch.matmul(single_vc_activations, kernels.detach())
            feature = torch.permute(feature, (2, 0, 1))
            features[k, :, :, :] = feature
        return features

    def manipulate_content(self, content, vmf_activations, vc_activations):
        with torch.no_grad():
            old_content = content.detach()
            old_vmf_activations = vmf_activations.detach()
            old_vc_activations = vc_activations.detach()

            new_content = torch.ones_like(old_content)
            new_content = new_content.to(self.device)

            resize_order = 1.5

            new_rv_resize = TVF.resize(old_content[:, 2, :, :].unsqueeze(1), (int(resize_order*old_content.size(2)), int(resize_order*old_content.size(3))), interpolation=InterpolationMode.NEAREST)
            new_rv = TVF.center_crop(new_rv_resize, (old_content.size(2), old_content.size(3)))
            new_content[:,2,:,:] = new_rv[:,0,:,:]
            new_content[:, 0:2, :, :] = old_content[:, 0:2, :, :]
            new_content[:, 3, :, :] = new_content[:,3,:,:] - old_content[:, 0, :, :]
            new_content[:, 3, :, :] = new_content[:,3,:,:] - old_content[:, 1, :, :]
            new_content[:, 3, :, :] = new_content[:, 3, :, :] - old_content[:, 2, :, :]

        return new_content

    def manipulate_content(self, content, vmf_activations, vc_activations):
        with torch.no_grad():
            old_content = content.detach()
            old_vmf_activations = vmf_activations.detach()
            old_vc_activations = vc_activations.detach()

            resize_order = 2

            new_lv_resize = TVF.resize(old_content[:, 0, :, :].unsqueeze(1), (int(resize_order*old_content.size(2)), int(resize_order*old_content.size(3))), interpolation=InterpolationMode.NEAREST)
            new_lv = TVF.center_crop(new_lv_resize, (old_content.size(2), old_content.size(3)))
            new_lv = (new_lv > 0.5) * 1.0


            mid_vmf_activations_resize = TVF.resize(old_vmf_activations, (
                int(resize_order * old_vmf_activations.size(2)), int(resize_order * old_vmf_activations.size(3))),
                                                       interpolation=InterpolationMode.NEAREST)
            mid_vmf_activations = TVF.center_crop(mid_vmf_activations_resize,
                                                     (old_vmf_activations.size(2), old_vmf_activations.size(3)))

            new_lv_expand= new_lv.expand(mid_vmf_activations.size())
            new_vmf_activations = new_lv_expand*mid_vmf_activations + (1-new_lv_expand)*old_vmf_activations


            norm_vmf_activations = torch.zeros_like(old_vmf_activations)
            norm_vmf_activations = norm_vmf_activations.to(self.device)
            for i in range(old_vmf_activations.size(0)):
                norm_vmf_activations[i, :, :, :] = F.normalize(new_vmf_activations[i, :, :, :], p=1, dim=0)
            condition = torch.zeros(old_vmf_activations.size(0), 1, old_vmf_activations.size(2), old_vmf_activations.size(3))
            condition = condition.to(self.device)
            new_content = self.content_iterater(norm_vmf_activations, condition)
            new_content = F.softmax(new_content, dim=1)
            # new_content = new_content > 0.5
            new_content = new_content.detach()

            new_vc_activations = torch.zeros_like(vc_activations)
            new_vc_activations = new_vc_activations.to(self.device)

            for i in range(old_content.size(0)):
                new_lv_vc_activations_resize = TVF.resize(old_vc_activations[i, :, :, :].unsqueeze(0), (int(resize_order*old_vc_activations.size(2)), int(resize_order*old_vc_activations.size(3))), interpolation=InterpolationMode.NEAREST)
                new_lv_vc_activations = TVF.center_crop(new_lv_vc_activations_resize, (old_vc_activations.size(2), old_vc_activations.size(3)))
                new_lv_vc_activations.squeeze(0)
                new_lv_mask = new_content[i, 0, :, :].expand(new_lv_vc_activations.size())
                new_lv_mask = new_lv_mask.detach()
                new_lv_mask = (new_lv_mask > 0.5)* 1.0
                new_lv_vc_activations = new_lv_vc_activations * new_lv_mask
                new_vc_activations[i, :, :, :] = new_lv_vc_activations + old_vc_activations[i, :, :, :]*(1-new_lv_mask)
                #
                # new_lv_vmf_activations_resize = TVF.resize(old_vmf_activations[i, :, :, :].unsqueeze(0), (
                # int(resize_order* old_vmf_activations.size(2)), int(resize_order* old_vmf_activations.size(3))), interpolation=InterpolationMode.NEAREST)
                # new_lv_vmf_activations = TVF.center_crop(new_lv_vmf_activations_resize, (old_vmf_activations.size(2), old_vmf_activations.size(3)))
                # new_lv_vmf_activations.squeeze(0)
                # new_lv_vmf_activations = new_lv_vmf_activations * new_lv_mask
                # new_vmf_activations[i, :, :, :] = new_lv_vmf_activations + old_vmf_activations[i, :, :, :]*(1-new_lv_mask)

            new_decoding_features = self.compose(new_content.detach(), new_vc_activations.detach(), new_vmf_activations.detach())
            new_rec = self.decoder(new_decoding_features.detach())
        return new_rec, new_content[:,0,:,:].unsqueeze(1)

    def manipulate_style(self, content, vmf_activations, vc_activations):
        with torch.no_grad():
            old_content = content.detach()
            old_vc_activations = vc_activations.detach()

            lv_mask = old_content[:, 0, :, :].unsqueeze(1)
            lv_mask = (lv_mask > 0.5) * 1.0
            lv_mask = lv_mask.expand(old_vc_activations.size())

            not_lv_mask = 1 - lv_mask

            lv_weight = self.content_iterater.conv1.weight
            lv_attention = lv_weight[:, :-1, :, :]
            # lv_attention = (lv_attention>0)*lv_attention
            # generate noise here:
            # lv_attention = lv_attention.expand(old_vc_activations.size())
            lv_features = old_vc_activations
            lv_std = torch.abs(old_vc_activations)/5

            lv_noise = torch.normal(mean=lv_features, std=lv_std)

            # new_vc_activations = old_vc_activations*lv_mask*0.1 + old_vc_activations*not_lv_mask
            new_vc_activations = lv_noise*lv_mask + old_vc_activations*not_lv_mask

            new_decoding_features = self.compose(old_content, new_vc_activations.detach(), vmf_activations.detach())
            new_rec = self.decoder(new_decoding_features.detach())

        return new_rec, lv_weight

    def content_regression(self, rec, content):
        reg_content = None
        return reg_content

    def visualise_L(self):
        vmf_activations = self.vc_activations
        norm_vmf_activations = vmf_activations
        lv_weight = self.content_iterater.conv1.weight[0,:-1,0,0]
        myo_weight = self.content_iterater.conv2.weight[0,:-1,0,0]
        rv_weight = self.content_iterater.conv3.weight[0,:-1,0,0]
        bg_weight = self.content_iterater.conv4.weight[0,:-1,0,0]

        # norm_vmf_activations = torch.zeros_like(vmf_activations)
        # norm_vmf_activations = norm_vmf_activations.to(self.device)
        # for i in range(vmf_activations.size(0)):
        #     norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)

        max_lv_weight, max_lv_weight_index = torch.max(lv_weight, dim=0)
        max_myo_weight, max_myo_weight_index = torch.max(myo_weight, dim=0)
        max_rv_weight, max_rv_weight_index = torch.max(rv_weight, dim=0)
        max_bg_weight, max_bg_weight_index = torch.max(bg_weight, dim=0)

        lv_heatimages = torch.zeros(vmf_activations.size(0), 3, vmf_activations.size(2), vmf_activations.size(3))
        myo_heatimages = torch.zeros(vmf_activations.size(0), 3, vmf_activations.size(2), vmf_activations.size(3))
        rv_heatimages = torch.zeros(vmf_activations.size(0), 3, vmf_activations.size(2), vmf_activations.size(3))
        bg_heatimages = torch.zeros(vmf_activations.size(0), 3, vmf_activations.size(2), vmf_activations.size(3))
        random_bg_heatimages = torch.zeros(vmf_activations.size(0), 3, vmf_activations.size(2), vmf_activations.size(3))

        # random_index = 0
        while 1:
            random_index = random.randint(0, 511)
            if bg_weight[random_index] > 0.2*max_bg_weight:
                break
        # random_index = random.randint(0, 511)
        # print(random_index)
        # print(max_bg_weight_index)

        for i in range(norm_vmf_activations.size(0)):
            lv_channel = norm_vmf_activations[i, int(max_lv_weight_index), :, :]
            myo_channel = norm_vmf_activations[i, int(max_myo_weight_index), :, :]
            rv_channel = norm_vmf_activations[i, int(max_rv_weight_index), :, :]
            bg_channel = norm_vmf_activations[i, int(max_bg_weight_index), :, :]

            image = lv_channel.detach().cpu().numpy() * 255
            image = image.astype(np.uint8)
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            heatmap = heatmap.transpose(2, 0, 1)
            heatmap = torch.from_numpy(heatmap).float() / 255.0
            lv_heatimages[i, :, :, :] = heatmap

            image = myo_channel.detach().cpu().numpy() * 255
            image = image.astype(np.uint8)
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            heatmap = heatmap.transpose(2, 0, 1)
            heatmap = torch.from_numpy(heatmap).float() / 255.0
            myo_heatimages[i, :, :, :] = heatmap

            image = rv_channel.detach().cpu().numpy() * 255
            image = image.astype(np.uint8)
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            heatmap = heatmap.transpose(2, 0, 1)
            heatmap = torch.from_numpy(heatmap).float() / 255.0
            rv_heatimages[i, :, :, :] = heatmap

            image = bg_channel.detach().cpu().numpy() * 255
            image = image.astype(np.uint8)
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            heatmap = heatmap.transpose(2, 0, 1)
            heatmap = torch.from_numpy(heatmap).float() / 255.0
            bg_heatimages[i, :, :, :] = heatmap

            random_bg_channel = norm_vmf_activations[i, int(random_index), :, :]
            image = random_bg_channel.detach().cpu().numpy() * 255
            image = image.astype(np.uint8)
            heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            heatmap = heatmap.transpose(2, 0, 1)
            heatmap = torch.from_numpy(heatmap).float() / 255.0
            random_bg_heatimages[i, :, :, :] = heatmap
        return [lv_heatimages, myo_heatimages, rv_heatimages, bg_heatimages, random_bg_heatimages]