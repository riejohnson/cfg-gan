There is no comprehensive documentation for this code.  I'm assuming that usage of this code is limited to, for example, either reproducing the xICFG experiments in our paper [Johnson+Zhang,ICML2018] or using it as a reference for developing your own implementation in some convenient platform.  And so the minimum information for these purposes is given here. 

### Image file format

Image files used as input to training should consist of a 16-byte header and pixels of all the images.  An image file for MNIST is provided, and one for SVHN can be downloaded by a script at `test/`.  The files for LSUN datasets are not provided as they are so large.  If needed, please make them as follows. 

__**Header**__  
  offset 0: number of channels  
  offset 4: width of each image  
  offset 8: height of each image  
  offset 12: number of images  
  
__**Pixels**__  
Each value of pixels should be represented by one byte (0x00-0xFF), and the pixels of the second image should immediately follow the pixels of the first image, and so forth.  The order within an image must be the same as in [ppm](http://netpbm.sourceforge.net/doc/ppm.html) or [pgm](http://netpbm.sourceforge.net/doc/pgm.html) files.  

That is, an image file should be a 16-byte header immediately followed by the pixel components of ppm/pgm files concatenated.  
  
__**File naming conventions**__  
A dataset can consist of more than one image file.  When a dataset consists of one image file (i.e., one batch), the filename should end with ".xbin".  When a dataset consists of, for example, 10 image files (i.e., 10 batches), the files should be named as _name_.xbin.1of10, _name_.xbin.2of10, ..., and _name_.xbin.10of10 (replacing _name_ with whatever you like), and `num_batches=10` should be specified in the training parameters.  
  
In our experiments, we made 10 batches (each with 130K bedrooms and 130K living rooms) for LSUN BR+LR and 
7 batches (each with 100K towers and 100K bridges) for LSUN T+B, and it is so assumed in the scripts at `test/`.  
  
### Interfaces

The network configuration parameters for the discriminator and approximator are generated by functions in `func.sh`.  The discriminator parameters start with `_d_`, and the approximator parameters start with `_g_`.  `_d_` or `_g_` is followed by a layer id, which is a number starting with 0.  For example, `_d_0layer_type=Act` means that layer\#0 of the discriminator is an activation layer.  Connection between layers is specified by `conn`, e.g., `_g_conn=0-1-2-3-top,1-3`.  Some of the network parameters are explained in "3.1.2 Layer parameters" of [this](http://riejohnson.com/software/conText-v4-ug.pdf)  (for text categorization), and the rest can be figured out, hopefully.  Other types of parameters such as `cfg_T` are self-explanatory.  When training is performed (by `test/cfg-dcganx.sh`, `test/cfg-fc2.sh`, etc.), the parameters are shown at the beginning of a log file at `test/log`. 

### Source code

In case you need a reference for developing your own implementation of xICFG, relevant source code files are at `bin/src/img/`.  Look for the method `cfg_train` in `AzpG_Cfg.cpp` there.  Other directories under `bin/src` are for general purposes and should be ignored.  

Please be warned that it may be hard to read as I didn't write this for reference purposes.
