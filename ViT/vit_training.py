import sys
sys.path.append('/workspace/transformers/src')
from transformers import ViTForImageClassification
import torch
from dataset import *
import argparse
from pathlib import Path 
import torch.utils.data as data
import torch.optim as optim
import wandb
from datetime import datetime
import pytz
import evaluate
from tqdm import tqdm
import albumentations


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description="Train params")
    
    parser.add_argument("--data_path", type=Path, default=Path(f"{ROOT}/data"))
    parser.add_argument("--model_path", type=str, default='google/vit-base-patch16-224')
    parser.add_argument("--save_path", type=Path, default=Path(f"{ROOT}/results"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    
    
    args = parser.parse_args()
    wandb.config.update(args)
    
    transform_train=albumentations.Compose([
            #albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albumentations.Resize(height = 224, width = 224),
            albumentations.pytorch.transforms.ToTensorV2()
            ])

    
    train_data = Dataset(args.data_path, "train", transform_train)
    val_data = Dataset(args.data_path, "val", transform_train)
    
    id2label = get_id2label(args.data_path, "train")

    train_dataloader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dataloader = data.DataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=0, shuffle=False)
    
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=args.model_path, 
        id2label=id2label, 
        num_labels=len(id2label),
        ignore_mismatched_sizes=True
        ).to(device)
    
    model_parameter = model.parameters()
    
    optimizer = optim.Adam(model_parameter, lr=args.lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=20)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    epochs = args.epochs
    
    best_score = 0
    accuracy = evaluate.load("accuracy")
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for batch in tqdm(train_dataloader):
            
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            
            output = model(image)
            
            loss = loss_fn(output.logits, label)
            
            loss.backward()
            total_loss += loss.detach().float()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            #print every batches
            wandb.log({"train_loss": loss, "lr": scheduler.get_last_lr()[0]})
            
        
        model.eval()
        pred_list, label_list = [], []
        for batch in tqdm(val_dataloader):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            with torch.no_grad():
                output = model(image)
            loss = loss_fn(output.logits, label)
            total_val_loss += loss.detach().float()
            
            logits = output.logits
            pred = logits.argmax(-1)  #model predicts one of the classes
            
            pred = pred.to("cpu").numpy()
            label = label.to("cpu").numpy()
            
            pred_list.append(pred)
            label_list.append(label)
            
            # pred_class = []
            # for idx in pred:
            #     pred_class.append(id2label[str(idx)])
            # print("Step:", step, "Predicted classes:", pred_class)
            
            score_step = accuracy.compute(predictions=pred, references=label)  #step단위
            
            wandb.log({"val_loss": loss, "accuracy": score_step['accuracy']})
            
        
        pred = np.concatenate(pred_list)
        label = np.concatenate(label_list)
            
        score = accuracy.compute(predictions=pred, references=label)
            
        if best_score < score['accuracy']:
            now = datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y%m%d-%H:%M:%S')
            Path(f"{args.save_path}/{now}").mkdir(parents=True, exist_ok=True)
            moodel_path = Path(f"{args.save_path}/{now}/weight_{epoch}.pt")
            optim_path = Path(f"{args.save_path}/{now}/optim_{epoch}.pt")
            torch.save(model.state_dict(), moodel_path)
            torch.save(optimizer.state_dict(), optim_path)
            best_score = score['accuracy']
        
        print(f" Epoch ({epoch}), Best accuracy: {best_score}")   
    
    
    

if __name__ == "__main__":
    ROOT =Path(__file__).parents[0]
    
    wandb.init(
        entity='yazzzing-korea-university',
        project='CV_study',
        name=datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y%m%d-%H:%M:%S')
    )
    
    main()


