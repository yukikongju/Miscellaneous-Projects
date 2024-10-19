Prerequisites:
- [X] Download pretrained model and place it into '/Users/emulie/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
- [ ] Redeem Google Cloud Credits and setup to train on GPU

Exercices:

- [X] Using a trained model on different image size
    * Load a trained model. Use the model on an image with the same input shape
    * Use the model on different shapes:
	+ (1,3,8,8)
	+ (1,3,64,64)
	+ (1,1,16,16)
    * We suggest using VGG19
- Adapting a trained model for a custom tasks
    * Use an already trained model for an alternative tasks
    * Try to use a UNet for:
	+ classification: VGG19 on CalTech101 Dataset (originally ImageNet)
	+ image segmentation: Cloud Dataset / MRI
	+ image augmentation: goblins dataset
	+ Style Transfer: Use Eman's images with prompt
- Implement Paper Architecture
- Fine-Tuning a pretrained classification model to add another sound

**Useful Links**

- [ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
- [Torchvision VGG19 model Link](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
- [Image Segmentation From Scratch in PyTorch](https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch)

**Kaggle Competitions**

Image Segmentaion:
- [Kaggle - Understanding Clouds from Satellite Images](https://www.kaggle.com/competitions/understanding_cloud_organization/data)


