import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis, RelativeAverageSpectralError, SpectralDistortionIndex
from torchmetrics.regression import MeanSquaredError
from torchinfo import summary

from model.CBT import *
from data_loader.DataLoader import GaoFen2, Sev2Mod, WV3
from utils import *
import numpy as np


def main(args):
    config_file = args.config
    # try:
    with open(get_config_path() / config_file, 'r') as file:
        config_data = yaml.safe_load(file)

        # input pipeline
        tr_dataset = eval(config_data['data_pipeline']
                          ['train']['dataset'])  # str to object
        tr_path = Path(config_data['data_pipeline']['train']['path'])
        tr_mslr_size = config_data['data_pipeline']['train']['mslr_img_size']
        tr_pan_size = config_data['data_pipeline']['train']['pan_img_size']

        tr_augmentation_list = []
        tr_shuffle = config_data['data_pipeline']['train']['preprocessing']['shuffle']
        tr_cropping_on_the_fly = config_data['data_pipeline']['train']['preprocessing']['cropping_on_the_fly']
        if config_data['data_pipeline']['train']['preprocessing']['RandomHorizontalFlip']['enable']:
            tr_augmentation_list.append(RandomHorizontalFlip(
                p=config_data['data_pipeline']['train']['preprocessing']['RandomHorizontalFlip']['prob']))
        if config_data['data_pipeline']['train']['preprocessing']['RandomVerticalFlip']['enable']:
            tr_augmentation_list.append(RandomVerticalFlip(
                p=config_data['data_pipeline']['train']['preprocessing']['RandomVerticalFlip']['prob']))
        if config_data['data_pipeline']['train']['preprocessing']['RandomRotation']['enable']:
            tr_augmentation_list.append(RandomRotation(
                degrees=config_data['data_pipeline']['train']['preprocessing']['RandomRotation']['degrees']))

        val_dataset = eval(
            config_data['data_pipeline']['validation']['dataset'])
        val_path = Path(config_data['data_pipeline']['validation']['path'])
        val_mslr_size = config_data['data_pipeline']['validation']['mslr_img_size']
        val_pan_size = config_data['data_pipeline']['validation']['pan_img_size']
        val_shuffle = config_data['data_pipeline']['validation']['preprocessing']['shuffle']
        val_cropping_on_the_fly = config_data['data_pipeline']['validation']['preprocessing']['cropping_on_the_fly']
        val_steps = config_data['data_pipeline']['validation']['val_steps']

        test_dataset = eval(config_data['data_pipeline']['test']['dataset'])
        test_path = Path(config_data['data_pipeline']['test']['path'])
        test_mslr_size = config_data['data_pipeline']['test']['mslr_img_size']
        test_pan_size = config_data['data_pipeline']['test']['pan_img_size']
        test_cropping_on_the_fly = config_data['data_pipeline']['test']['preprocessing']['cropping_on_the_fly']

        # general settings
        model_name = config_data['general_settings']['name']
        model_type = eval(config_data['general_settings']['model_type'])
        continue_from_checkpoint = True
        checkpoint_name = config_data['general_settings']['checkpoint_name']
        if checkpoint_name:
            checkpoint_path = get_checkpoint_path() / model_name / checkpoint_name

        # task
        upscale = config_data['task']['upscale']
        mslr_to_pan_scale = config_data['task']['mslr_to_pan_scale']

        # network configs
        patch_size = config_data['network']['patch_size']
        in_chans = config_data['network']['in_chans']
        embed_dim = config_data['network']['embed_dim']
        depths = config_data['network']['depths']
        num_heads = config_data['network']['num_heads']
        window_size = config_data['network']['window_size']
        compress_ratio = config_data['network']['compress_ratio']
        squeeze_factor = config_data['network']['squeeze_factor']
        conv_scale = config_data['network']['conv_scale']
        overlap_ratio = config_data['network']['overlap_ratio']
        mlp_ratio = config_data['network']['mlp_ratio']
        qkv_bias = config_data['network']['qkv_bias']
        qk_scale = config_data['network']['qk_scale']
        drop_rate = config_data['network']['drop_rate']
        attn_drop_rate = config_data['network']['attn_drop_rate']
        drop_path_rate = config_data['network']['drop_path_rate']
        norm_layer = eval(config_data['network']['norm_layer'])
        ape = config_data['network']['ape']
        patch_norm = config_data['network']['patch_norm']
        img_range = config_data['network']['img_range']
        upsampler = config_data['network']['upsampler']
        resi_connection = config_data['network']['resi_connection']
        hab_wav = config_data['network']['hab_wav']
        scbab_wav = config_data['network']['scbab_wav']
        ocab_wav = config_data['network']['ocab_wav']
        ocbab_wav = config_data['network']['ocbab_wav']

        # training_settings
        batch_size = config_data['training_settings']['batch_size']
        optimizer_type = eval(
            config_data['training_settings']['optimizer']['type'])
        learning_rate = config_data['training_settings']['optimizer']['learning_rate']
        betas = config_data['training_settings']['optimizer']['betas']
        lr_decay_type = eval(
            config_data['training_settings']['scheduler']['type'])
        lr_gamma = config_data['training_settings']['scheduler']['gamma']
        loss_type = eval(config_data['training_settings']['loss']['type'])

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu" # if not enough available memory
    print(device)

    # Initialize DataLoader
    train_dataset = tr_dataset(
        tr_path, transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=tr_shuffle, drop_last=True)

    validation_dataset = val_dataset(
        val_path)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=val_shuffle)

    te_dataset = test_dataset(
        test_path)
    test_loader = DataLoader(
        dataset=te_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = model_type(pan_img_size=(tr_pan_size[0], tr_pan_size[1]), pan_low_size_ratio=mslr_to_pan_scale, patch_size=patch_size, in_chans=in_chans,
                       embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, compress_ratio=compress_ratio,
                       squeeze_factor=squeeze_factor, conv_scale=conv_scale, overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                       ape=ape, patch_norm=patch_norm, upscale=upscale, img_range=img_range, upsampler=upsampler, resi_connection=resi_connection,
                       hab_wav = hab_wav, scbab_wav = scbab_wav, ocab_wav = ocab_wav, ocbab_wav = ocbab_wav,
                       mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                       pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = optimizer_type(
        model.parameters(), lr=learning_rate, betas=(betas[0], betas[1]))
    criterion = loss_type().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
        'rase' : RelativeAverageSpectralError().to(device),
        'mse' : MeanSquaredError().to(device),
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
        'rase' : RelativeAverageSpectralError().to(device),
        'mse' : MeanSquaredError().to(device),
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device),
        'sam': SpectralAngleMapper().to(device),
        'ergas': ErrorRelativeGlobalDimensionlessSynthesis().to(device),
        'rase' : RelativeAverageSpectralError().to(device),
        'mse' : MeanSquaredError().to(device),
    })

    sdi_metric = SpectralDistortionIndex().to(device)
    sdi_results = []

    val_report_loss = 0
    test_report_loss = 0
    highest_psnr = 0
    highest_ssim = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []

    ergas_l = 4


    ergas_score = 0
    sam_score = 0
    q2n_score = 0
    psnr_score = 0
    ssim_score = 0

    # Model summary
    summary(model, [(1, 1, test_pan_size[0], test_pan_size[1]), (1, in_chans, test_mslr_size[0], test_mslr_size[1])],
            dtypes=[torch.float32, torch.float32], depth=12)

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics = load_checkpoint(torch.load(
            checkpoint_path), model, optimizer, tr_metrics, val_metrics)

    choose_dataset = str(config_data['data_pipeline']
                         ['train']['dataset'])

    # evaluation mode
    model.eval()
    with torch.no_grad():
        test_iterator = iter(test_loader)
        for i, (pan, mslr, mshr) in enumerate(test_iterator):
            # forward
            pan, mslr, mshr = pan.to(device), mslr.to(
                device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            # Normalize preds and target for SDI
            # print(mssr.max())
            preds_normalized = mssr / mssr.max()
            target_normalized = mshr / mshr.max()

            # Calculate SDI on normalized predictions and targets
            sdi_value = sdi_metric(preds_normalized, target_normalized)
            # print(sdi_value)
            sdi_results.append(sdi_value.item())

            figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
            axis[0].imshow((scaleMinMax(mslr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :1], cmap='viridis')
            axis[0].set_title('(a) LR')
            axis[0].axis("off")

            axis[1].imshow(pan.permute(0, 3, 2, 1).detach().cpu()[
                0, ...], cmap='gray')
            axis[1].set_title('(b) PAN')
            axis[1].axis("off")

            axis[2].imshow((scaleMinMax(mssr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :1], cmap='viridis')
            axis[2].set_title(
                f'(c) CBT {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
            axis[2].axis("off")

            axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :1], cmap='viridis')
            axis[3].set_title('(d) GT')
            axis[3].axis("off")

            plt.savefig(f'results/Images_{choose_dataset}_{i}.png')

            mslr = mslr.permute(0, 3, 2, 1).detach().cpu().numpy()
            pan = pan.permute(0, 3, 2, 1).detach().cpu().numpy()
            mssr = mssr.permute(0, 3, 2, 1).detach().cpu().numpy()
            gt = mshr.permute(0, 3, 2, 1).detach().cpu().numpy()

            np.savez(f'results/img_array_{choose_dataset}_{i}_{model_name}.npz', mslr=mslr,
                     pan=pan, mssr=mssr, gt=gt)
            

        # compute metrics
        test_metric = test_metric_collection.compute()
        test_metric_collection.reset()

        # Compute the average SDI
        average_sdi = sum(sdi_results) / len(sdi_results)

        # Print final scores
        print(f"Final scores:\n"
              f"ERGAS: {test_metric['ergas'].item()}\n"
              f"SAM: {test_metric['sam'].item()}\n"
              f"PSNR: {test_metric['psnr'].item()}\n"
              f"SSIM: {test_metric['ssim'].item()}\n"
              f"RASE: {test_metric['rase'].item()}\n"
              f"MSE: {test_metric['mse'].item()}\n"
              f"D_lambda: {average_sdi:.4f}")


def scaleMinMax(x):
    return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of CrossFormer')
    parser.add_argument('-c', '--config', type=str,
                        help='config file name', required=True)

    args = parser.parse_args()

    main(args)
