# DeepLearningFinalProject
Coco2014 image captioning with visual attention

**Ensure that you are in the `scratch` folder.**\
The code below dumps torch binary features and binary vocab dictionary.
```
git clone https://github.com/garyongguanjie/DeepLearningFinalProject.git
cd DeepLearningFinalProject
python build_vocab.py
```
```
wget -c "https://sutdapac-my.sharepoint.com/:u:/g/personal/gary_ong_mymail_sutd_edu_sg/EUE8VQN6j7dNrRyhPLoCVFkBXYyRoQgcicrRQM_PhxYslg?e=xS0idk&download=1" -O weights.zip
unzip weights.zip
```
## TODOS
* Training loop
* Check correctness. Not 100% sure if got mistakes. There might be some misalignment between inputs and outputs?
* Evaluation loop
* Output words
* Bleu Score
* Frontend
### Extras if got time
* Beam search
* Plot attention and show on frontend (definitely many points for this)
