import torch.nn as nn


class ResnetFeatures(nn.Module):
    def __init__(self,resnet):
        """
        args 
        resnet -> any resnet from torchvision.models
        """
        super().__init__()
        self.model = nn.Sequential(*list(resnet.children())[:-2])#chop off avg pool and fc layers
    
    def forward(self,x):
        """
        produces features = layer layer before avgpooling
        
        bs x n x h x w resize to vector form i.e. bs x n x h*w 
        """
        x = self.model(x)
        bs,c,h,w = x.shape
        return x.view(bs,-1,h*w)

# if __name__ == '__main__':
#     from MyCocoDataset import MyCocoCaptions
#     import config
#     from torchvision import transforms
#     from torchvision import models
#     transforms =  transforms.Compose([transforms.Resize((224,224)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                                     ])

#     train_dataset = MyCocoCaptions(root = config.TRAIN_IMG_PATH,
#                         annFile = config.TRAIN_JSON_PATH,
#                         transform=transforms)
    
#     tensor,captions,image_name = train_dataset[0]

#     resnet34 = models.resnet34(pretrained=True)
#     FeatureExtractor = ResnetFeatures(resnet34)
#     FeatureExtractor.eval()
#     features = FeatureExtractor(tensor.unsqueeze(0))
#     print(features.shape)
#     print(features)