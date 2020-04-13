from MyCocoDataset import MyCocoCaptions
from ResnetFeatureExtractor import ResnetFeatures
import config
from torchvision import transforms
from torchvision import models
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def make_feature_dump(dataloader,model,device,directory):
    """
    passes images into model and dumps feature pickles into directory
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)
    with torch.no_grad():
        model.to(device)
        model.eval()
        for data in tqdm(dataloader):
            img,captions,filenames = data
            img = img.to(device)
            outputs = model(img)
            for i in range(len(filenames)):
                image = outputs[i].cpu()
                filename = filenames[i][:-4] + '.pth'
                filename = os.path.join(directory,filename)
                torch.save(image,filename)
                
if __name__ == "__main__":
    transforms =  transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    train_dataset = MyCocoCaptions(root = config.TRAIN_IMG_PATH,
                        annFile = config.TRAIN_JSON_PATH,
                        transform=transforms)

    val_dataset = MyCocoCaptions(root=config.VAL_IMG_PATH,
                            annFile = config.VAL_JSON_PATH,
                            transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=256,num_workers=4)

    resnet34 = models.resnet34(pretrained=True)
    
    FeatureExtractor = ResnetFeatures(resnet34)
    FeatureExtractor.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    make_feature_dump(train_loader,FeatureExtractor,device,'./train_features')
    make_feature_dump(val_loader,FeatureExtractor,device,'./val_features')

