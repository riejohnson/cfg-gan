#!/bin/bash
#####  Generate 10K images using a saved model and 
#####  evaluate the quality using a classifier. 
#####  input:  a model file, a classifier for evaluating images. 
#####  output: a score in stdout

  fnm=cfg-gen-eval

  gpu=-1  # gpu#.  -1 means "use default"

  source func.sh
  gpumem=${gpu}:8  # pre-allocate 5GB device memory

  #---  To use a pre-trained model, uncomment "nm=svhn" or "nm=brlr". 
  #---  Otherwise, change "nm" and "model_fn" to point your model at mod/
  #---  e.g., "nm=mnist; model_fn=my-mnist.ddg"
  nm=svhn
#  nm=brlr

  model_fn=${nm}-conv.ddg

  #---  Download a pre-trained model. ---
  _fn=$model_fn
  cd $moddir
  _download  # download training data 
  if [ $? != 0 ]; then echo $fnm: failed to download the pre-trained model.; exit 1; fi
  cd .. 
  #--------------------------------------

  cls_fn=${clsdir}/${nm}-cls.ReNet
  gen_fn=${gendir}/${nm}-conv

  mb=64
  random_seed=17

  #---  Generate 10k images in the *.xbin format. 
  $exe $gpumem cfg_gen Bin model_fn=${moddir}/${model_fn} num_gen=10000 \
             gen_fn=$gen_fn mini_batch_size=$mb gen_seed=$random_seed
  if [ $? != 0 ]; then echo $fnm: generation failed.; exit 1; fi


  #---  Evaluate the generated images in the *.xbin format 
  #---  using a classifier. 
  $exe $gpumem eval_img classif_fn=$cls_fn \
             tstname=$gen_fn datatype=imgbin x_ext=.xbin Scale-1+1 \
             mini_batch_size=$mb
  if [ $? != 0 ]; then echo $fnm: evalimg failed.; exit 1; fi

