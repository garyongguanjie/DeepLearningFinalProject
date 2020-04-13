# DeepLearningFinalProject
Coco2014 image captioning with visual attention\

## Dump feature maps into files
This is done without finetuning. As 1 epoch on the forward pass takes about ~10 minutes.
Ensure that you are in the `scratch` folder.
```
git clone https://github.com/garyongguanjie/DeepLearningFinalProject.git
cd DeepLearningFinalProject
python makeFeatureDump.py
```
