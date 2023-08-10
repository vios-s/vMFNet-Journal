import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import logging
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TVF
import metrics.gan_loss
from metrics.focal_loss import FocalLoss
from torch.utils.data import DataLoader, random_split
import utils
from eval import eval_compcsd
from loaders.mms_dataloader_dg_aug import get_dg_data_loaders
import models
from composition.losses import ClusterLoss
import losses
from torch.utils.tensorboard import SummaryWriter

def get_args():
    usage_text = (
        "CompCSD Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=50, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type= int, default=4, help='Number of inputs per batch')
    parser.add_argument('-c', '--cp', type=str, default='checkpoints', help='The name of the checkpoints.')
    parser.add_argument('-t', '--tv', type=str, default='D', help='The name of the checkpoints.')
    parser.add_argument('-w', '--wc', type=str, default='SDNet_LR00002_nB_FT', help='The name of the checkpoints.')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-enc', '--encoder_dir', type=str, default='cp_unet_100_tvA/', help='The name of the pretrained encoder checkpoints.')
    parser.add_argument('-mn','--model_name', type=str, default='compcsd2', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')

    return parser.parse_args()

# python train_un.py -e 200 -bs 4 -c cp_compcsd2_tvA_un/ -enc cp_unet_100_tvA/UNet.pth -t A -w CompCSD_12un_tvA2 -g 0



lr_patience = 4
layer = 8
vc_num = 12 # kernel numbers

def latent_norm(a):
    n_batch, n_channel, _, _ = a.size()
    for batch in range(n_batch):
        for channel in range(n_channel):
            a_min = a[batch,channel,:,:].min()
            a_max = a[batch, channel, :, :].max()
            a[batch,channel,:,:] -= a_min
            a[batch, channel, :, :] /= a_max - a_min
    return a

def train_net(args):
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    dir_checkpoint = args.cp
    test_vendor = args.tv
    wc = args.wc
    enc_dir = args.encoder_dir

    #Model selection and initialization
    model_params = {
        'image_channels': 1,
        'layer': layer,
        'vc_numbers': vc_num,
        'num_classes': 3,
        'anatomy_out_channels': 4,
        'z_length': 8,
        'vMF_kappa': 30
    }
    model = models.get_model(args.model_name, model_params)
    num_params = utils.count_parameters(model)
    # print(model)
    print('Model Parameters: ', num_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)
    #################################################### load pre-trained encoder and vMF kernels
    model.load_encoder_weights(enc_dir, device)
    if layer == 6:
        kernels_save_dir = test_vendor + '8_' + str(vc_num) + 'kernels/'
    elif layer == 7:
        kernels_save_dir = test_vendor + '4_' + str(vc_num) + 'kernels/'
    elif layer == 8:
        kernels_save_dir = test_vendor + '2_' + str(vc_num) + 'kernels/'
    else:
        kernels_save_dir = test_vendor + '_' + str(vc_num) + 'kernels/'
    init_path = kernels_save_dir + 'init/'
    kernel_save_name = 'dictionary_'+str(vc_num)+'.pickle'
    dict_dir = init_path + 'dictionary/'+kernel_save_name
    model.load_vmf_kernels(dict_dir)
    # models.initialize_weights(model, args.weight_init)
    #################################################### load pre-trained encoder and vMF kernels

    train_labeled_loader, train_labeled_dataset, train_unlabeled_loader, train_unlabeled_dataset, test_loader, test_dataset = get_dg_data_loaders(args.batch_size, test_vendor=test_vendor, image_size=224)

    n_val = int(len(train_labeled_dataset) * 0.1)
    n_train = len(train_labeled_dataset) - n_val

    train, val = random_split(train_labeled_dataset, [n_train, n_val])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)
    train_loader = train_labeled_loader
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)
    val_loader = train_labeled_loader

    print(len(train))
    print(len(val))
    print(len(train_unlabeled_dataset))

    #metrics initialization
    l2_distance = nn.MSELoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    l1_distance = nn.L1Loss().to(device)
    focal = FocalLoss()
    cluster_loss = ClusterLoss()

    # discriminator = models.get_dis(model_params)
    # num_params = utils.count_parameters(discriminator)
    # print('Discriminator Parameters: ', num_params)
    # models.initialize_weights(discriminator, args.weight_init)
    # discriminator.to(device)

    # #optimizer initialization
    # dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
    # # need to use a more useful lr_scheduler
    # dis_scheduler = optim.lr_scheduler.StepLR(optimizer=dis_optimizer, step_size=20)

    #optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=lr_patience)

    writer = SummaryWriter(comment=wc)

    global_step = 0

    for epoch in range(epochs):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, true_masks in train_loader:
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                ce_mask = true_masks.clone().to(device=device, dtype=torch.long)
                true_masks = true_masks.to(device=device, dtype=mask_type)
                rec, pre_seg, content, features, kernels, L_visuals = model(imgs, layer=layer)

                clu_loss = cluster_loss(features.detach(), kernels)

                batch_loss = clu_loss

                pbar.set_postfix(**{'loss (batch)': batch_loss.item()})

                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar('loss/batch_loss', batch_loss.item(), global_step)
                writer.add_scalar('loss/cluster_loss', clu_loss.item(), global_step)

                if global_step % ((n_train//batch_size) // 2) == 0:
                    a_out = content
                    writer.add_images('images/train', imgs, global_step)
                    writer.add_images('Latent/a_out0', a_out[:,0,:,:].unsqueeze(1), global_step)
                    writer.add_images('Latent/a_out1', a_out[:, 1, :, :].unsqueeze(1), global_step)
                    writer.add_images('Latent/a_out2', a_out[:, 2, :, :].unsqueeze(1), global_step)
                    writer.add_images('Latent/a_out3', a_out[:, 3, :, :].unsqueeze(1), global_step)
                    writer.add_images('images/train_reco', rec, global_step)
                    writer.add_images('images/train_true', true_masks[:, 0:3, :, :], global_step)
                    writer.add_images('images/train_pred', pre_seg[:, 0:3, :, :] > 0.5, global_step)
                    writer.add_images('L_visuals/L_1', L_visuals[:,0,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_2', L_visuals[:,1,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_3', L_visuals[:,2,:,:].unsqueeze(1), global_step)
                    writer.add_images('L_visuals/L_4', L_visuals[:,3,:,:].unsqueeze(1), global_step)
                    if vc_num > 4:
                        writer.add_images('L_visuals/L_5', L_visuals[:,4,:,:].unsqueeze(1), global_step)
                    if vc_num > 5:
                        writer.add_images('L_visuals/L_6', L_visuals[:,5,:,:].unsqueeze(1), global_step)
                    if vc_num > 6:
                        writer.add_images('L_visuals/L_7', L_visuals[:,6,:,:].unsqueeze(1), global_step)
                    if vc_num > 7:
                        writer.add_images('L_visuals/L_8', L_visuals[:,7,:,:].unsqueeze(1), global_step)
                    if vc_num > 8:
                        writer.add_images('L_visuals/L_9', L_visuals[:,8,:,:].unsqueeze(1), global_step)
                    if vc_num > 9:
                        writer.add_images('L_visuals/L_10', L_visuals[:,9,:,:].unsqueeze(1), global_step)
                    if vc_num > 10:
                        writer.add_images('L_visuals/L_11', L_visuals[:,10,:,:].unsqueeze(1), global_step)
                    if vc_num > 11:
                        writer.add_images('L_visuals/L_12', L_visuals[:,11,:,:].unsqueeze(1), global_step)

                pbar.update(imgs.shape[0])

                global_step += 1


    try:
        os.mkdir(dir_checkpoint)
        logging.info('Created checkpoint directory')
    except OSError:
        pass
    torch.save(model.state_dict(),
               dir_checkpoint + 'CP_epoch.pth')
    logging.info('Checkpoint saved !')

    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)

    train_net(args)