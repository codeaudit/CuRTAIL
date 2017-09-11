# CurTAIL

This repository provides an API for CurTAIL framework (http://arxiv.org/abs/1709.02538). CurTAIL can be used as a generic framework for characterizing and thwarting  adversarial samples in the context of deep learning.

The repo includes an example with mnist dataset. To evaluate the framework for different applications, you'll need to slightly modify the files.



## python packages required:
Tensorflow(v 1.2.1), Keras(v 2.0.6), scipy(v 0.19.1), sklearn(v 0.18.2)

## matlab packages required:
ompbox10. See http://www.cs.technion.ac.il/~ronrubin/software.html for installation instructions.




## Description of the files:

**model_architecture.py:** includes a function "this_model" where you define the network using Keras.

**read_dataset.py:** read the dataset, convert to floating point if necessary, reshape, etc.

**train_model.py:** train a baseline neural network and save the model.

**train_MRR_module.py:** trains a latent dictionary module at a certain checkpoint layer.

**train_dictionary.py:** learns a dictionary for denoising input images.

**denoiser.m:** a matlab script that uses ompbox10 to denoise images.

**create_adv.py:** create adversarial samples using the specified attack algorithm.

**curtail.py:** includes modules required for all attack and defense algorithms.





## Steps to use the framework:

Here we go through the steps for using CuRTAIL to evaluate over the MNIST dataset. You can skip steps 0-3 as the output files are already availabel in this repo but you should run steps 4-6 to generate the corresponding files and evaluate CuRTAIL.

**Step-0:** specify neural network architecture in the "model_architecture.py" file if you wish to evaluate CuRTAIL on another model. Then, modify the "read_dataset.py" if you which to use another dataset. 

**Step-1:** train your baseline neural network using keras with Tensorflow backend, and save the trained model. An example is provided in "train_model.py". On top of the file, specify "path_to_save" to save the trained model. Here we saved the model in 'mnist_baseline', so you do not need to re-train the model from scratch.

**Step-2:** train your latent dictionary using the "train_MRR_module.py" file. On top of the file specify the following:
      
      path_to_baseline: this is the model you trained in step 1
      path_to_save-detector: this will be the output of this file
      parameters for training: batch_size, number of epochs, learning rate, weight decay, momentum, and parameter gamma (see Equation 11 in the paper).
      checkpoint_id: the id of the checkpoint layer in "model.layers" where "model" is the baseline keras model.

**Step-3:** train dictionaries for each class using the "train_dictionary.py" file. specify the following parameters on top of the file:
      
      path_to_save_dictionaries: the path to save the learned dictionaries
      height, width, and depth of the images
      patch_size: this is the size of the window of neighbor pixels used for learning the dictionary
      dic_size: the number of columns in each dictionary. Same as the dimensionality of sparse codes

**Step-4:** generate adversarial samples using the "create_adv.py" module. On top of the file, specify the following:
      
      victim_model_path: this is the path to the main model you trained in Step-1
      method: this is the method to generate adversarial samples. possible values are 'Deepfool', 'FSG', and 'JSMA'
      niters: number of iterations in the pertinent attack. This is a list of integer numbers.
      epsilons: the epsilon value used in the attack. This is a list of real number in the range (0-1)
      batch_size: number of images that are run in parallel (for speed-up purposes)
      path_to_save_samples: the directory where you wish to save the adversarial samples.
      
**Step-5:** denoise the adversarial images using the matlab script "denoiser.m". makesure you have installed the ompbox10 module and added the installation path to Matlab directory. specify the following on top of the file:
      
      omp_K: the number of nonzero elements in each sparse code. 
      patch_size: must be compatible with the one specified in Step-3
      adv-dir: the parent directory of adversarial samples. must be compatible with the one specified in Step-4
      method: the method for which you wish to denoise the adversarial images.
      niters: number of iterations (As in Step-4) 
      epsilons: Epsilon values (As in Step-4)
      path_to_dict: path to the learned dictionaries (specified in Step-3)
  
  
 **Step-6:** run the "detect_adversarial.py" file to check the detection rate of the module. Specify the following on top of the file:
      
      path_to_detector: the detector latent dictionary module trained in Step-2
      path_to_baseline: the main (victim) model trained in Step-1
      path_to_images: the denoised images along with the original adversarial images outputted in Step-5
      checkpint_id: the id of the checkpoint layer as in Step-2 (must be compatible with Step-2)
      SP: Security Parameter as defined in the paper
      
      
      
      

