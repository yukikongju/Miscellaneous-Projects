Prerequisites:
- [X] Download pretrained model and place it into '/Users/emulie/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
- [ ] Redeem Google Cloud Credits and setup to train on GPU

Exercices:

- Using a trained model on different image size
    * Load a trained model. Use the model on an image with the same input shape
    * Use the model on different shapes:
	+ (1,3,8,8)
	+ (1,3,64,64)
	+ (1,1,16,16)
    * We suggest using VGG19
- Adapting a trained model for a custom tasks
    * Use an already trained model for an alternative tasks
    * Try to use a UNet for (1) image segmentation (2) image augmentation (3) classification
- Implement Paper Architecture
- Fine-Tuning a pretrained classification model to add another sound

**Useful Links**

- [ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)

[Torchvision VGG19 model Link](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)


