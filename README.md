© MIT 6.S191: [Introduction to Deep Learning](http://introtodeeplearning.com) 

# Face Detection
The model will be able to classify images as either faces or not faces. Hence I need face images (positive) and non-face images (negative). I will minimize the bias in data.
## Datasets
* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A large-scale (over 200K images) of celebrity faces.   
* [ImageNet](http://www.image-net.org/). Many images across many different categories.
## Variational autoencoder (VAE) to generate data with desired feature 

![The concept of a VAE](/img/vae.jpg)

VAEs rely on an encoder-decoder structure to learn a latent representation of the input data. The encoder network takes in input images, 
encodes them into a series of variables defined by a mean and standard deviation, 
and then draws from the distributions defined by these parameters to generate a set of sampled latent variables. 
The decoder network then "decodes" these variables to generate a reconstruction of the original image, 
which is used during training to help the model identify which latent variables are important to learn. 

### Understanding VAEs: loss function

In practice, how can we train a VAE? In learning the latent space, we constrain the means and standard deviations to approximately follow a unit Gaussian. Recall that these are learned parameters, and therefore must factor into the loss computation, and that the decoder portion of the VAE is using these parameters to output a reconstruction that should closely match the input image, which also must factor into the loss. What this means is that we'll have two terms in our VAE loss function:

1. **Latent loss ($L_{KL}$)**: measures how closely the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence.
2. **Reconstruction loss ($L_{x}{(x,\hat{x})}$)**: measures how accurately the reconstructed outputs match the input and is given by the $L^1$ norm of the input image and its reconstructed output.  

The equations for both of these losses are provided below:

$$ L_{KL}(\mu, \sigma) = \frac{1}{2}\sum\limits_{j=0}^{k-1}\small{(\sigma_j + \mu_j^2 - 1 - \log{\sigma_j})} $$

$$ L_{x}{(x,\hat{x})} = ||x-\hat{x}||_1 $$ 

Thus for the VAE loss we have: 

$$ L_{VAE} = c\cdot L_{KL} + L_{x}{(x,\hat{x})} $$

where $c$ is a weighting coefficient used for regularization. 




