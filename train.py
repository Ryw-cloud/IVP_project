from model import Net, Discriminator
import torch
from torch.utils.data import DataLoader
from dataloader import AniTriplet
import numpy as np
import argparse
from datetime import datetime
import os
import time
import pickle


def save_stats(save_dir, exp_time, hyperparams, stats):
    save_path = os.path.join(save_dir, exp_time)
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(os.path.join(save_path, 'hyperparams.pickle')):
        with open(os.path.join(save_path, 'hyperparams.pickle'), 'wb') as handle:
            pickle.dump(hyperparams, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()
    with open(os.path.join(save_path, 'stats.pickle'), 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_stats_path', type=str, required=True)
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--max_num_images', default=None)
    parser.add_argument('--save_model_path', required=True)
    args = parser.parse_args()

    # process information to save statistics
    hyperparams = {
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'eval_every': args.eval_every,
        'max_num_images': args.max_num_images
    }

    # instantiate models, loss functions and optimisers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)
    discriminator = Discriminator()
    discriminator.weight_init(mean=0.0, std=0.02)
    discriminator = discriminator.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    mse_loss = torch.nn.MSELoss()
    mse_loss.to(device)
    bce_loss = torch.nn.BCELoss()
    bce_loss.to(device)

    # to store evaluation metrics
    train_loss = []
    val_loss = []
    current_best_val_loss = float('inf')

    # build dataloaders
    print('Building train/val dataloaders...')
    seq_dir = os.path.join(args.dataset_path, 'sequences')
    train_txt = os.path.join(args.dataset_path, 'tri_trainlist.txt')
    trainset = AniTriplet(args.dataset_path)
    n = len(trainset)
    print("N", n)
    n_train = int(n * 0.8)
    n_val = n - n_train
    
    trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print('Train/val dataloaders successfully built!')

    # start training
    print('\nTraining...')
    for epoch in range(args.num_epochs):
        num_batches = 0
        train_loss_epoch = [0, 0, 0]
        
        model.train()
        discriminator.train()

        # for time calculations
        start_time = time.time()
        if args.max_num_images is not None:
            train_batches = int(np.ceil(float(args.max_num_images) / args.batch_size))
        else:
            train_batches = len(trainloader)
        
        for i in trainloader:
            # load data
            
            first = i['first_last_frames'][0]
            #print(first.shape)
            last = i['first_last_frames'][1]
            mid = i['middle_frame']
            first, last, mid = first.to(device), last.to(device), mid.to(device)

            mid_recon, flow_t_0, flow_t_1, w1, w2 = model(first, last)

            # Add discriminator 
            d_optimizer.zero_grad()
            d_real_result = discriminator(first, mid, last)
            # Prevent generator backward pass, to reduce computation time
            d_fake_result = discriminator(first, mid_recon.detach(), last)
            d_loss_real = 0.5 * bce_loss(d_real_result, torch.ones_like(d_real_result).to(device))
            d_loss_fake = 0.5 * bce_loss(d_fake_result, torch.zeros_like(d_fake_result).to(device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # training
            optimizer.zero_grad()
            d_fake_result = discriminator(first, mid_recon, last)
            loss =  0.999 * mse_loss(mid, mid_recon) + 0.001 * bce_loss(d_fake_result, torch.ones_like(d_fake_result).to(device))
            loss.backward()
            optimizer.step()

            # store stats       
            train_loss_epoch[0] += loss.item()
            train_loss_epoch[1] += d_loss_real.item()
            train_loss_epoch[2] += d_loss_fake.item()
            num_batches += 1

            if args.max_num_images is not None:
                if num_batches == train_batches:
                    break

            
        train_loss_epoch[0] /= num_batches
        train_loss_epoch[1] /= num_batches
        train_loss_epoch[2] /= num_batches
        print('Epoch [{} / {}] Train g_loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(epoch+1, args.num_epochs, train_loss_epoch[0], train_loss_epoch[1], train_loss_epoch[2]))

        # for evaluation, save best model and statistics
        if epoch % args.eval_every == 0:
            train_loss.append([0,0,0])
            val_loss.append([0,0,0])
            train_loss[-1] = train_loss_epoch

            model.eval()
            discriminator.eval()

            start_time = time.time()
            val_batches = len(valloader)

            with torch.no_grad():
                num_batches = 0
                for i in valloader:
                    first = i['first_last_frames'][0]
                    last = i['first_last_frames'][1]
                    mid = i['middle_frame']
                    first, last, mid = first.to(device), last.to(device), mid.to(device)

                    mid_recon, _, _, _, _ = model(first, last)
                    d_fake_result = discriminator(first, mid_recon, last)
                    loss = g_loss = 0.99 * mse_loss(mid, mid_recon) + 0.01 * bce_loss(d_fake_result, torch.ones_like(d_fake_result).to(device))
                    d_real_result = discriminator(first, mid, last)
                    d_loss_real = 0.5 * bce_loss(d_real_result, torch.ones_like(d_real_result).to(device))
                    d_loss_fake = 0.5 * bce_loss(d_fake_result, torch.zeros_like(d_fake_result).to(device))

                    # store stats
                    val_loss[-1][0] += loss.item()
                    val_loss[-1][1] += d_loss_real.item()
                    val_loss[-1][2] += d_loss_fake.item()
                    num_batches += 1


                val_loss[-1][0] /= num_batches
                val_loss[-1][1] /= num_batches
                val_loss[-1][2] /= num_batches
                print('Val g_loss: {}, d_loss_real: {}, d_loss_fake: {}'.format(val_loss[-1][0], val_loss[-1][1], val_loss[-1][2]))

                # save best model so far, according to validation loss
                if val_loss[-1][0] < current_best_val_loss:
                    current_best_val_loss = val_loss[-1][0]
                    torch.save(model, args.save_model_path)
                    print("Saved new best model!")

            # save statistics
            stats = {
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            save_stats(args.save_stats_path, hyperparams, stats)
            print("Saved stats!")