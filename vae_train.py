import torch
import torch.nn as nn
from unet_scatch import UNet
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pathlib

from vae import Vae

def main():
    parser = argparse.ArgumentParser(description='Argparse for train params')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--input_list', type=list, default=[(3, 64), (64, 128), (128, 256), (256, 512)])
    parser.add_argument('--output_list', type=list, default=[(1024, 512), (512, 256), (256, 128), (128, 64)])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_train_epoch', type=int, default=100)
    parser.add_argument('--path', type=pathlib.Path, default=pathlib.Path(f"{REPO_PATH}/result"))

    # args 에 위의 내용 저장
    args = parser.parse_args()
    #train_data = 
    #test_data = 

    model = Vae(args.input_list, args.output_list)
    
    model_parameter = model.parameters()
    
    optimizer = optim.Adam(model_parameter, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-08, weight_decay=0)
    loss_fn = nn.MSELoss()
    schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    epochs =args.num_train_epoch

    best_score = 10000
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            
            input = batch['input']
            target = batch['target']
            
            recon, z, mu, std, var, logvar = model(input)
            
            recon_loss = loss_fn(recon, target)
            kl_loss = args.delta * torch.mean(
                torch.pow(mu, 2) + var - 1.0 - logvar, dim=1
            )
            
            loss = recon_loss + kl_loss
            
            loss.backward()
            total_loss += loss.detach().float()
            optimizer.step()
            schedular.step()
            optimizer.zero_grad()
        
        model.eval()
        for step, batch in enumerate(test_dataloader):
            input = batch['input']
            target = batch['target']
            with torch.no_grad():
                output = model.inference(input)
            loss = loss_fn(output, target)
            total_val_loss += loss.detach().float()
            
        if best_score > total_val_loss:
            torch.save(model.state_dict(), args.path)
            torch.save(optimizer.state_dict(), args.path)
            best_score = total_val_loss
            


if __name__ == '__main__':
    REPO_PATH = "/home/yeajin/CV_study"

    # 인자값을 받을 수 있는 인스턴스 생성

    main()
    