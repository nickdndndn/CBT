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
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchinfo import summary

from model.CBT import *
from data_loader.DataLoader import GaoFen2, Sev2Mod, WV3
from utils import *


def main(args):
    config_file = args.config
    
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
        continue_from_checkpoint = config_data['general_settings']['continue_from_checkpoint']
        checkpoint_name = config_data['general_settings']['checkpoint_name']
        if checkpoint_name:
            checkpoint_path = get_checkpoint_path() / model_name / checkpoint_name
        report_interval = config_data['general_settings']['report_interval']
        save_interval = config_data['general_settings']['save_interval']
        evaluation_interval = config_data['general_settings']['evaluation_interval']
        test_intervals = config_data['general_settings']['test_intervals']

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
        steps = config_data['training_settings']['steps']
        batch_size = config_data['training_settings']['batch_size']
        optimizer_type = eval(
            config_data['training_settings']['optimizer']['type'])
        learning_rate = config_data['training_settings']['optimizer']['learning_rate']
        betas = config_data['training_settings']['optimizer']['betas']
        lr_decay_type = eval(
            config_data['training_settings']['scheduler']['type'])
        lr_decay_intervals = config_data['training_settings']['scheduler']['lr_decay_intervals']
        lr_gamma = config_data['training_settings']['scheduler']['gamma']
        loss_type = eval(config_data['training_settings']['loss']['type'])

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Initialize DataLoader
    train_dataset = tr_dataset(
        tr_path, transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=tr_shuffle, drop_last=True)  # , collate_fn=collate_fn

    validation_dataset = val_dataset(
        val_path)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=val_shuffle)  # , collate_fn=collate_fn

    test_dataset = test_dataset(
        test_path)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)  # , collate_fn=collate_fn

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

    scheduler = lr_decay_type(optimizer, step_size=1, gamma=lr_gamma)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Model summary
    summary(model, [(1, 1, test_pan_size[0], test_pan_size[1]), (1, in_chans, test_mslr_size[0], test_mslr_size[1])],
            dtypes=[torch.float32, torch.float32], depth=16)

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics = load_checkpoint(torch.load(
            checkpoint_path), model, optimizer, tr_metrics, val_metrics)

    print('==> Starting training ...')

    train_iter = iter(train_loader)
    train_progress_bar = tqdm(iter(range(49999, steps)), total=steps, desc="Training",
                              leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
    for step in train_progress_bar:
        if step % save_interval == 0 and step != 0:
            checkpoint = {'step': step, #step
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'tr_metrics': tr_metrics,
                          'val_metrics': val_metrics}
            save_checkpoint(checkpoint, model_name, current_daytime)

        try:
            # Samples the batch
            pan, mslr, mshr = next(train_iter)
        except:  # StopIteration
            # restart the loader if the previous loader is exhausted.
            train_iter = iter(train_loader)
            pan, mslr, mshr = next(train_iter)

        # forward
        pan, mslr, mshr = pan.to(device), mslr.to(device), mshr.to(device)
        mssr = model(pan, mslr)
        tr_loss = criterion(mssr, mshr)
        tr_report_loss += tr_loss
        batch_metric = metric_collection.forward(mssr, mshr)

        # backward
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # lr_decay step
        if step in lr_decay_intervals:
            scheduler.step()
            # print(scheduler.get_last_lr())

        batch_metrics = {'loss': tr_loss.item(),
                         'psnr': batch_metric['psnr'].item(),
                         'ssim': batch_metric['ssim'].item()}

        # report metrics
        train_progress_bar.set_postfix(
            loss=batch_metrics["loss"], psnr=f'{batch_metrics["psnr"]:.4f}', ssim=f'{batch_metrics["ssim"]:.4f}')

        # Store metrics
        if (step + 1) % report_interval == 0 and step != 0:
            # Batch metrics
            tr_report_loss = tr_report_loss / (report_interval)
            tr_metric = metric_collection.compute()

            # store metrics
            tr_metrics.append({'loss': tr_report_loss.item(),
                               'psnr': tr_metric['psnr'].item(),
                               'ssim': tr_metric['ssim'].item()})

            # reset metrics
            tr_report_loss = 0
            metric_collection.reset()

        # Evaluate model
        if (step + 1) in evaluation_interval and step != 0:
            # evaluation mode
            model.eval()
            with torch.no_grad():
                print("\n==> Start evaluating ...")
                val_steps = val_steps if val_steps else len(validation_loader)
                eval_progress_bar = tqdm(iter(range(val_steps)), total=val_steps, desc="Validation",
                                         leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
                val_iter = iter(validation_loader)
                for eval_step in eval_progress_bar:
                    try:
                        # Samples the batch
                        pan, mslr, mshr = next(val_iter)
                    except StopIteration:
                        # restart the loader if the previous loader is exhausted.
                        val_iter = iter(validation_loader)
                        pan, mslr, mshr = next(val_iter)
                    # forward
                    pan, mslr, mshr = pan.to(device), mslr.to(
                        device), mshr.to(device)
                    mssr = model(pan, mslr)
                    val_loss = criterion(mssr, mshr)
                    val_metric = val_metric_collection.forward(mssr, mshr)
                    val_report_loss += val_loss

                    # report metrics
                    eval_progress_bar.set_postfix(
                        loss=f'{val_loss.item()}', psnr=f'{val_metric["psnr"].item():.2f}', ssim=f'{val_metric["ssim"].item():.2f}')

                # compute metrics total
                val_report_loss = val_report_loss / len(validation_loader)
                val_metric = val_metric_collection.compute()
                val_metrics.append({'loss': val_report_loss.item(),
                                    'psnr': val_metric['psnr'].item(),
                                    'ssim': val_metric['ssim'].item()})

                print(
                    f'\nEvaluation: avg_loss = {val_report_loss.item():.4f} , avg_psnr= {val_metric["psnr"]:.4f}, avg_ssim={val_metric["ssim"]:.4f}')

                # reset metrics
                val_report_loss = 0
                val_metric_collection.reset()
                print("==> End evaluating <==\n")

            # train mode
            model.train()

        # test model
        if (step + 1) in test_intervals and step != 0:
            # evaluation mode
            model.eval()
            with torch.no_grad():
                print("\n==> Start testing ...")
                test_progress_bar = tqdm(iter(test_loader), total=len(
                    test_loader), desc="Testing", leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
                for pan, mslr, mshr in test_progress_bar:
                    # forward
                    pan, mslr, mshr = pan.to(device), mslr.to(
                        device), mshr.to(device)
                    mssr = model(pan, mslr)
                    test_loss = criterion(mssr, mshr)
                    test_metric = test_metric_collection.forward(mssr, mshr)
                    test_report_loss += test_loss

                    # report metrics
                    test_progress_bar.set_postfix(
                        loss=f'{test_loss.item()}', psnr=f'{test_metric["psnr"].item():.2f}', ssim=f'{test_metric["ssim"].item():.2f}')

                # compute metrics total
                test_report_loss = test_report_loss / len(test_loader)
                test_metric = test_metric_collection.compute()
                test_metrics.append({'loss': test_report_loss.item(),
                                     'psnr': test_metric['psnr'].item(),
                                     'ssim': test_metric['ssim'].item()})

                print(
                    f'\nTesting: avg_loss = {test_report_loss.item():.4f} , avg_psnr= {test_metric["psnr"]:.4f}, avg_ssim={test_metric["ssim"]:.4f}')

                # reset metrics
                test_report_loss = 0
                test_metric_collection.reset()
                print("==> End testing <==\n")

            # train mode
            model.train()

            # save best test model based on PSNR
            if test_metrics[-1]['psnr'] > best_test_psnr:
                best_test_psnr = test_metrics[-1]['psnr']
                checkpoint = {'step': step,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'tr_metrics': tr_metrics,
                              # 'val_metrics': val_metrics,
                              'test_metrics': test_metrics}
                save_checkpoint(checkpoint, model_name,
                                current_daytime + '_best_test')

    print('==> training ended <==')

    # test model
    model.eval()
    with torch.no_grad():
        print("\n==> Start testing ...")
        test_progress_bar = tqdm(iter(test_loader), total=len(
            test_loader), desc="Testing", leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
        for pan, mslr, mshr in test_progress_bar:
            # forward
            pan, mslr, mshr = pan.to(device), mslr.to(
                device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            # report metrics
            test_progress_bar.set_postfix(
                loss=f'{test_loss.item()}', psnr=f'{test_metric["psnr"].item():.2f}', ssim=f'{test_metric["ssim"].item():.2f}')

        # compute metrics total
        test_report_loss = test_report_loss / len(test_loader)
        test_metric = test_metric_collection.compute()
        test_metrics.append({'loss': test_report_loss.item(),
                             'psnr': test_metric['psnr'].item(),
                             'ssim': test_metric['ssim'].item()})

        print(
            f'\nTesting: avg_loss = {test_report_loss.item():.4f} , avg_psnr= {test_metric["psnr"]:.4f}, avg_ssim={test_metric["ssim"]:.4f}')

        # reset metrics
        test_report_loss = 0
        test_metric_collection.reset()
        print("==> End testing <==\n")


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Original Implementation of CBT in Pytorch')
    parser.add_argument('-c', '--config', type=str,
                        help='config file name', required=True)

    args = parser.parse_args()

    main(args)
