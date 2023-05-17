import torch
from torch.utils.data import DataLoader
from dataloader import AnimeDataSet, AniTriplet
import argparse
import os
import numpy as np
import skimage.metrics
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--saved_model_path', type=str, required=True)
    args = parser.parse_args()

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.saved_model_path, map_location=torch.device(device))
    model = model.to(device)
    model.eval()

    # build test dataloader
    print('Building test dataloader...')
    seq_dir = os.path.join(args.dataset_path, 'sequences')
    test_txt = os.path.join(args.dataset_path, 'tri_testlist.txt')
    testset = AnimeDataSet(args.dataset_path)
    #testset  = VimeoDataset(seq_dir, test_txt)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    print('Test dataloader successfully built!')

    # get evaluation metrics PSNR and SSIM for test set
    print('\nTesting...')
    with torch.no_grad():
        psnr = 0
        ssim = 0
        num_samples = len(testloader)
        start_time = time.time()
        cnt = 0

        for i in testloader:
            first = i['first_last_frames'][0]
            last = i['first_last_frames'][1]
            mid = i['middle_frame']
            first, last, mid = first.to(device), last.to(device), mid.to(device)

            mid_recon, _, _, _, _ = model(first, last)

            mid = mid.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))
            mid_recon = mid_recon.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))

            # print(mid.shape)
            # print(mid_recon.shape)
            # calculate PSNR and SSIM
            psnr += skimage.metrics.peak_signal_noise_ratio(mid, mid_recon, data_range=1)
            ssim += skimage.metrics.structural_similarity(mid, mid_recon, data_range=1, multichannel=True, channel_axis =2 )

        psnr /= num_samples
        ssim /= num_samples
        print('Test set PSNR: {}, SSIM: {}'.format(psnr, ssim))