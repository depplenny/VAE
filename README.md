© MIT 6.S191: [Introduction to Deep Learning](http://introtodeeplearning.com) 

# Face Detection
The model will be able to classify images as either faces or not faces. Hence I need face images (positive) and non-face images (negative). I will minimize the bias in data.
## Datasets
* [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A large-scale (over 200K images) of celebrity faces.   
* [ImageNet](http://www.image-net.org/). Many images across many different categories.
## Variational autoencoder (VAE) to generate data with desired feature 

![](https://deeplearningfromscratch.files.wordpress.com/2018/07/vae_complete.png?w=1140)


Encoder outputs mean and  diagonal covariance vector
![\mu=\mu(x,\phi), \quad \Sigma=\Sigma(x,\phi)=diag(\sigma_1^2, \sigma_2^2, ...),](https://render.githubusercontent.com/render/math?math=%5Cmu%3D%5Cmu(x%2C%5Cphi)%2C%20%5Cquad%20%5CSigma%3D%5CSigma(x%2C%5Cphi)%3Ddiag(%5Csigma_1%5E2%2C%20%5Csigma_2%5E2%2C%20...)%2C)
Then a latent vector is calculated with the help of unit Gaussian distribution
$$z=\mu+\Sigma^{1/2} \odot \epsilon,$$
Decder reconstructs the input
$$\hat{x}=\hat{x}(z,\theta),$$
where $\phi$ and $\theta$ are trainable weights.

For this network, we define the loss as

1. **Latent loss ($L_{KL}$)**: measures how closely the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence,
$$L_{KL} = \frac{1}{2}\sum\limits_{j=0}^{k-1}\small{(\sigma_j + \mu_j^2 - 1 - \log{\sigma_j})}.$$
2. **Reconstruction loss ($L_x$)**: measures how accurately the reconstructed outputs match the input and is given by the $L^1$ norm of the input image and its reconstructed output,
$$L_{x} = ||x-\hat{x}||_1.$$   


Thus for the VAE loss we have: 

$$L_{VAE} = c\cdot L_{KL} + L_{x}{(x,\hat{x})},$$

where $c$ is a weighting coefficient used for regularization. 


