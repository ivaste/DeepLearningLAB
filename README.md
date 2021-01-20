# DeepLearningLAB
MSc laboratory experiences and Homeworks at UNIPD 2020

# Homeworks

## Homework 1 - Supervised Deep Learning
**Notebook link 1:** [Homework 1 - Regression](https://github.com/ivaste/DeepLearningLAB/blob/main/Homework%201/HW1_IvancichStefano_1227846_Regression.ipynb)  
**Notebook link 2:** [Homework 1 - Classification](https://github.com/ivaste/DeepLearningLAB/blob/main/Homework%201/HW1_IvancichStefano_1227846_Classification.ipynb)  
**Report:** [HW1 Ivancich Stefano 1227846.pdf](https://github.com/ivaste/DeepLearningLAB/blob/main/Homework%201/HW1%20Ivancich%20Stefano%201227846.pdf)  
**Content:**
 - 2 pt: implement basic regression and classification tasks
 - 2 pt: explore advanced optimizers and regularization methods
 - 1 pt: optimize hyperparameters using grid/random search and cross-validation
 - 2 pt: implement CNN for classification task on MNIST
 - 1 pt: visualize weight histograms, activation profiles and receptive fields

## Homework 2 - Unsupervised Deep Learning
**Notebook link:** [Homework 2 - Unsupervised Deep Learning](https://github.com/ivaste/DeepLearningLAB/blob/main/Homework%202/HW2_IvancichStefano_1227846.ipynb)  
**Report:** [HW2 Ivancich Stefano 1227846.pdf](https://github.com/ivaste/DeepLearningLAB/blob/main/Homework%202/HW2%20Ivancich%20Stefano%201227846.pdf)  
**Content:**
 - 1 pt: implement and test (convolutional) autoencoder, reporting the trend of reconstruction loss and some examples of image reconstruction
 - 1 pt: explore advanced optimizers and regularization methods
 - 1 pt: optimize hyperparameters using grid/random search and cross-validation
 - 1 pt: explore the latent space structure (e.g., PCA, t-SNE) and generate new samples from latent codes
 - 1 pt: implement and test denoising (convolutional) autoencoder
 - 1 pt: fine-tune the (convolutional) autoencoder using a supervised classification task (you can compare classification accuracy and learning speed with results achieved in homework 1)
 - 2 pt: implement variational (convolutional) autoencoder or GAN

## Homework 3 - Deep Reinforcement Learning
**Notebook link:** [Homework 3 - Deep Reinforcement Learning](https://github.com/ivaste/DeepLearningLAB/blob/main/Homework%203/HW3_IvancichStefano_1227846.ipynb)  
**Report:** [...work in progress...]()  
**Content:**
 - 2 pt: extend the notebook used in Lab 07, in order to study how the exploration profile (either using eps-greedy or softmax) impacts the learning curve. Try to tune the model hyperparameters or tweak the reward function in order to speed-up learning convergence (i.e., reach the same accuracy with fewer training episodes).
 - 3 pt: extend the notebook used in Lab 07, in order to learn to control the CartPole environment using directly the screen pixels, rather than the compact state representation used during the Lab (cart position, cart velocity, pole angle, pole angular velocity). This will require to change the “observation_space”.
 - 3 pt: train a deep RL agent on a different Gym environment. You are free to choose whatever Gym environment you like from the available list, or even explore other simulation platforms.


# Laboratories

## Lab 2 - Introduction to PyTorch
**Notebook link:** [nndl_2020_lab_02_pytorch_intro](https://github.com/ivaste/DeepLearningLAB/blob/main/Lab%2002/nndl_2020_lab_02_pytorch_intro.ipynb)  
**Content:**
 - Basics: Tensors, Operations on GPU, Autograd
 - Network trainig procedure: initialization, process input, loss, backpropagation, Optimizer
 - Dataset and Dataloader
 
## Lab 3 - Regression and Classification with PyTorch
**Notebook link:** [nndl_2020_lab_03_regression](https://github.com/ivaste/DeepLearningLAB/blob/main/Lab%2003/nndl_2020_lab_03_regression.ipynb)  
**Content:**
 - Regression with 2 FC layers
 - Classification with 2FC layers
 - Access to network parameters of a Fully connected layer
 - Weights histogram
 - Analyze activations
 
## Lab 4 - Text Generation
**Notebook link:** [nndl_2020_lab_04_text_generation](https://github.com/ivaste/DeepLearningLAB/blob/main/Lab%2004/nndl_2020_lab_04_text_generation.ipynb)  
**Content:**
 - shakespeare sonnets
 - One hot encoding
 - Train LSTM
 - Generate text letter by letter
 
 ## Lab 5 - Convolutional Autoencoder
 **Notebook link:** [nndl_2020_lab_05_convolutional_autoencoder](https://github.com/ivaste/DeepLearningLAB/blob/main/Lab%2005/nndl_2020_lab_05_convolutional_autoencoder_with_solutions.ipynb)  
 **Content:**
 - Encoder-Decoder
 - Encoded Space Visualization
 - Generate samples from the encoded space 
 
 ## Lab 6 - Transfer Learning
 **Notebook link:** [nndl_2020_lab_06_transfer_learning](https://github.com/ivaste/DeepLearningLAB/blob/main/Lab%2006/nndl_2020_lab_06_transfer_learning_with_solutions.ipynb)  
 **Content:**
 - Alex net
 - Network analysis
 
 ## Lab 7 - Reinforcement Learning
 **Notebook link:** [nndl_2020_lab_07_deep_reinforcement_learning](https://github.com/ivaste/DeepLearningLAB/blob/main/Lab%2007/nndl_2020_lab_07_deep_reinforcement_learning_with_solutions.ipynb)  
 **Content:**
  - DQN
  - Gym Environment (CartPole-v1)
