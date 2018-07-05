# This repository is under construction. 

## CFG for GAN in C++
This is a C++ implementation of composite functional gradient learning of generative adversarial models, described in our ICML paper:  
* [Composite functional gradient learning of generative adversarial models](https://arxiv.org/abs/1801.06309).  Rie Johnson and Tong Zhang.  ICML 2018.  

**_System requirements_**  
To run this code, your system must have the following: 
* a CUDA-capable GPU with 12GB or larger device memory (e.g., Tesla P100). 
  (5GB device memory will do for small 32x32 images, though.) 
* CUDA 8.0 or higher. 
* Testing was done with gnu g++ and CUDA 8.0 on Linux.  
  I don't know how to run it on Windows though it should be possible.  

**_What you can do_**  
* Training of image generation models using the xICFG algorithm of the paper above.  
* Generation of images using a model trained with xICFG.   
* Evaluation of image quality, using classifiers (pre-trained and used in the paper). 

**_How_**  
1. Go to `bin/` and customize `makefile` there if needed; see the beginning of `makefile` for instructions.  
   Build executables by entering `make` at `bin/`. 
2. To test the executable, go to `test/` and enter `./test.sh`.  `Dreal-Dgen` should gradually go down if the executable was built correctly.  
3. Try out `*.sh` at `test/'.  

Here is slightly more [technical details](doc/info.md).  

**_FAQ_**  
Q. Do you have a tensorflow or pyTorch version of this code?  
A. No.  Please let me know if you've made one.  

Q. Is there a python wrapper for this code?  
A. No.  Instead, you have shell scripts (`test/*.sh`).  The network architecture and training/generation parameters can be changed by changing the shell scripts.
  
Q. Why C++?  
A. For historical reasons.  But don't worry, you don't have to read it if you just want to use this code. 

Q. I just want to see your xICFG implementation as a reference for developing my own code for doing xICFG.  Which source file should I look at?    
A. Please see "Source code" in [doc/info.md](doc/info.md).  

 
**_Data Source_**: The data files included here or downloaded by the scripts here 
were derived from [MNIST](http://yann.lecun.com/exdb/mnist/)
and [SVHN](http://ufldl.stanford.edu/housenumbers/).  
Note that the LSUN datasets used in the paper are not provided due to their sizes.  

**_License_**: This program is free software issued under the [GNU General Public License V3](http://www.gnu.org/copyleft/gpl.html). 

**_Note_**: This repository provides a snapshot of research code, which is constantly changing elsewhere for research purposes.  For this reason, it is very likely that pull requests 
(including typo corrections) will be declined. 
