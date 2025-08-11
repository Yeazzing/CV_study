from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json

#function to prepare data set....ONLY RUN ONCE
def prepare_data(datatype, data, data_root):
    if datatype == 'train':        
        Path(f"{data_root}/train").mkdir(parents=True, exist_ok=True)
    elif datatype == 'val':
        Path(f"{data_root}/val").mkdir(parents=True, exist_ok=True)
    
    Path(f"{data_root}/{datatype}/images").mkdir(parents=True, exist_ok=True)
    
    classes = data.features["label"].names
    
    labels = dict()
    for i in tqdm(range(len(data))):
        img = data[i]['image']
        label = data[i]['label']
        
        labels[label] = classes[label]
        img.save(Path(f'{data_root}/{datatype}/images/{i}-{classes[label]}.png'), 'png')
        
    file_path = Path(f'{data_root}/{datatype}/labels.txt')
    with open(file_path, 'w') as file:
        json.dump(labels, file,indent=4)
    print(f"Copying {datatype} data is done")

if __name__ == "__main__":
    ROOT =Path(__file__).parents[0]
    data_root = Path(f"{ROOT}/data")

    ds = load_dataset("ethz/food101")  #https://huggingface.co/datasets/ethz/food101 데이터 다운로드
    
    train_data = ds['train']
    val_data = ds['validation']
    
    prepare_data('train', train_data, data_root)
    prepare_data('val', val_data, data_root)    
            
            
            
            
            
    

    