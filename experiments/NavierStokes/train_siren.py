import os
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm

from Siren import Siren
from GridDataset import GridDataset
from NSLoss import NSLoss

assert torch.cuda.is_available()

def main(physics_coef: int, hidden_size: int, only_grid: bool, w0: int, wh: int, model_save_path: str):
    dataset = GridDataset("data/Case_01/measurements_flow/postProcessing_BL/winSpeedMapVector/",
                          "data/Case_01/measurements_turbines/30000_BL/rot_yaw_combined.csv",
                          "data/Case_01/winDir_processed.csv", only_grid_values=only_grid)

    MIN_TIME, MAX_TIME = dataset[0][0][:, 2][0].item(), dataset[-1][0][:, 2][0].item()

    train, tmp = random_split(dataset, [0.6, 0.4])
    val, _ = random_split(tmp, [0.5, 0.5])

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=4)

    in_features = 3 if only_grid else 35
    model = Siren(in_features=in_features, 
                  out_features=5,
                  hidden_features=hidden_size,
                  hidden_layers=5,
                  outermost_linear=True,
                  first_omega_0=w0,
                  hidden_omega_0=wh).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = NSLoss(physics_coef=physics_coef)

    epochs = 300

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)

    for epoch in tqdm(range(1, epochs+1)):
        running_train_losses = defaultdict(list)
        
        for inputs, targets in train_loader:
            model.train()

            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

            inputs[:, :2] = inputs[:, :2] / 299
            inputs[:, 2] = (inputs[:, 2] - MIN_TIME) / MAX_TIME
                
            optimizer.zero_grad()

            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            loss.backward()

            running_train_losses['total'].append(loss.item())
            running_train_losses['data'].append(data_loss.item())
            running_train_losses['physics'].append(physics_loss.item())
            
            optimizer.step()

        train_losses['total'].append(np.mean(running_train_losses['total']))
        train_losses['data'].append(np.mean(running_train_losses['data']))
        train_losses['physics'].append(np.mean(running_train_losses['physics']))

        running_val_losses = defaultdict(list)

        for inputs, targets in val_loader:
            model.eval()

            inputs = inputs.flatten(0, 1).float().cuda(non_blocking=True)
            targets = targets.flatten(0, 1).float().cuda(non_blocking=True)

            inputs[:, :2] = inputs[:, :2] / 299
            inputs[:, 2] = (inputs[:, 2] - MIN_TIME) / MAX_TIME

            outputs, input_coords = model(inputs)
            loss, data_loss, physics_loss = criterion(input_coords, outputs, targets)

            running_val_losses['total'].append(loss.item())
            running_val_losses['data'].append(data_loss.item())
            running_val_losses['physics'].append(physics_loss.item())

        val_losses['total'].append(np.mean(running_val_losses['total']))
        val_losses['data'].append(np.mean(running_val_losses['data']))
        val_losses['physics'].append(np.mean(running_val_losses['physics']))

        print(f"Epoch: {epoch} - Train Losses: {train_losses['total'][-1]:.4f}, {train_losses['physics'][-1]*physics_coef:.4f} - \
            Val Losses: {val_losses['total'][-1]:.4f}, {val_losses['physics'][-1]*physics_coef:.4f}")
        

        torch.save({
            'epoch': epoch,
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f"{model_save_path}/model{epoch}.pt")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train your SIREN model.")

    parser.add_argument("--physics", type=int, default=100, help="Physics loss multiplier")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--only-grid", type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument("--wi", type=int, default=30)
    parser.add_argument("--wh", type=int, default=30)

    args = parser.parse_args()

    model_save_path = f"models/Siren{args.hidden_size}-{args.wi}-{args.wh}"
    os.makedirs(model_save_path, exist_ok=True)
    
    main(args.physics, args.hidden_size, args.only_grid, args.wi, args.wh, model_save_path)