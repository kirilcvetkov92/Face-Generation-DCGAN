# DCGAN implementation in PyTorch

In this notebook, there is implemented GAN using convolutional layers in the generator and discriminator. This is called a Deep Convolutional GAN, or DCGAN for short. The DCGAN architecture was first explored in 2016 and has seen impressive results in generating new images; you can read the [original paper, here](https://arxiv.org/pdf/1511.06434.pdf).


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![Project Video](https://img.youtube.com/vi/_MNsEo0y8xk/0.jpg)](https://www.youtube.com/watch?v=_MNsEo0y8xk)


### Usage
1. Download Algined and Cropped CelebA dataset.
2. Use the [face_detect](https://github.com/sunshineatnoon/Paper-Implementations/blob/ad23812046ae8fa6c9c16fd26a8b1a14b4c10a59/BEGAN/Data/face_detect.py) script to crop images.
3. To train the model, run the main script (Check flags for other tunable options):


## Project notes
Defining the problems that I faced during the training : 
* Non-convergence: the model parameters oscillate, destabilize and never converge,
* Mode collapse: the generator collapses which produces limited varieties of samples,

The original data is biased and the background or the face position and perspective may cause some learning difficultes.
It required some preprocessing.  
I trained the model for 20 epochs nearly 10 hours of training with Adam optimizer and tried different network sizes before.  
It seems easier to train the model for 32x32 or 64x64 images, but with 128x128 I needed to be patient and to resolve some   problems.  
I experienced with Non-convergence/Mode collapse, and tried a lot of approaches to improve my model.  
    

* Started with : 
  ConvDim = 128  
  Gamma = 0.5  
  learning rate = 0.0002 and original dataset CelebA  

* Attempt 1 :  
Pros : It started generating something that looks like human  
Cons : I was not satisfied with the performance and I got Non-convergence/Mode collapse  after 10 epoch.  

* Attempt 2 : 
Preprocessed CelebA dataset with face detection (opencv haarcascade, same as I used for the previous project)  
Croped the face part picture to 128x128  

    * Pros : It started generating more realistic images
    * Cons : Still experienced Non-convergence/Mode collapse problems after some training period

* Attempt 3 : Changed Inverse convolution layer with Upsampling2D with NN interpolation  

    * Pros : It started generating more realistic images
    * Cons : Still experienced the Non-convergence/Model collapse problems.


Attempt 4 : 
- Decreased learning rate to 0.0001  
- Added learning rate decay, so the LR was decayed by 0.95 every epoch  

    * Pros : It started generating even more realistic images, the training was stabilized, I haven't experienced Non-convergence/Model collapse problems.
    * Cons : The training was slower but stable, keeping Generator lost from in safe range from 0.3 to 0.6 and distriminator 
