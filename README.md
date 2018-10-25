# Generative Adversarial Networks (GANs)
Vanilla  GAN and DCGAN

## Requirements
* Python 3.x
* Tensorflow > 0.12
* Numpy
* SciPy
* OpenCV
* lmdb (for processing LSUN dataset only)

## Pre-execution instructions
### Datasets to download
Download following files in the program root dirctory (*.../gan)
* LSUN : [download](https://github.com/fyu/lsun)

## Results
### Vanilla GAN
* Dataset class: LSUN/church outdoor
#### Generated sample:
<img src="https://user-images.githubusercontent.com/13673119/47503076-34fc8280-d8a4-11e8-8f80-4d8ad36253c5.png" width="300" height="300"></img>
#### Each epoch:
<img src="https://user-images.githubusercontent.com/13673119/47503496-3da18880-d8a5-11e8-9004-bdeccbe34265.GIF" width="500" height="800"></img>
#### Loss graph:
<img src="https://user-images.githubusercontent.com/13673119/47503532-514cef00-d8a5-11e8-98a7-18c68a9dc4aa.png" width="300" height="200"></img>

### DCGAN
* Dataset class: LSUN/church outdoor
#### Generated sample:
<img src="https://user-images.githubusercontent.com/13673119/47503141-5f4e4000-d8a4-11e8-9621-dbb81fd9be85.png" width="300" height="300"></img>
#### Each epoch:
<img src="https://user-images.githubusercontent.com/13673119/47503582-717cae00-d8a5-11e8-8309-957d65392b16.GIF" width="500" height="800"></img>
#### Loss graph:
<img src="https://user-images.githubusercontent.com/13673119/47503588-73df0800-d8a5-11e8-8055-65200bff0303.png" width="300" height="200"></img>

## Reference papers
*  I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative Adversarial Nets. NIPS 2014
 *  I. Goodfellow. NIPS 2016 Tutorial: Generative Adversarial Networks. NIPS 2016
 *  A. Radford, L. Metz, and S. Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR 2016
