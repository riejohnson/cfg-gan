#!/bin/bash
#####  Generate images using a saved model and make a collage.  
#####  input:  a model file: downloaded from the internet. 
#####          a classifier for sorting images. 
#####  output: a ppm/pgm file of a collage of images. 
#####          Use "convert" to convert it to JPEG etc. 

  gpu=-1 # GPU id.  -1: use default. 
  source func.sh
  fnm=cfg-gen-collage
  gpumem=${gpu}:8

  genseed=11 # Random seed.  Change this to get different images. 

  #---  To use a pre-trained model, uncomment "nm=brlr ..." or "nm=svhn ...". 
  #---  Otherwise, specify your model, for example:
  #---  "nm=mnist; modfn=my-mnist.ddg; imgext=pgm; numcls=10"
  #---      where the model file "my-mnist.ddg" is at mod/.  
#  nm=brlr;  modfn=${nm}-conv.ddg; imgext=ppm; numcls=2  # LSUN BR+LR
  nm=svhn;  modfn=${nm}-conv.ddg; imgext=ppm; numcls=10

  #---  Uncomment one of the following 5 lines 
  #---  See Appendix B of [Johnson+Zhang,ICML18] for the meaning of 'best' and 'worst'.  
  type=each-best    # Show 'best' images for each class
#  type=high-entropy # Show 'worst' images in terms of entropy.  
#  type=class0-best  # Use this for brlr to show bedrooms only
#  type=class1-best  # Use this for brlr to show living roomw only. 
#  type=random       # Show random sample. 

  echo; echo ---  $nm $type  ---

  #---  Download a pre-trained model. ---
  _fn=$modfn
  cd $moddir
  _download  # download training data 
  if [ $? != 0 ]; then echo $fnm: failed to download the pre-trained model.; exit 1; fi
  cd .. 
  #--------------------------------------

  cls_fn=${clsdir}/${nm}-cls.ReNet

  gap=0        # gap between images 
  ww=15; hh=10 # width and height of the collage in terms of the number of images

  mynm=${nm}-${ww}x${hh}-s${genseed}

  log_fn=${logdir}/${mynm}.glog

  #---  Load a saved model from a file and generate images in the xbin format. 
  opt=
  num_gen=$(( ww*hh*20 ))
  if   [ "$type" = "each-best"    ]; then gen_ppm_opt="num_each=$(( ww*hh/numcls )) num_gen=$num_gen"
  elif [ "$type" = "high-entropy" ]; then gen_ppm_opt="Entropy num_gen=$num_gen"
  elif [ "$type" = "class0-best"  ]; then gen_ppm_opt="class=0 num_gen=$num_gen"
  elif [ "$type" = "class1-best"  ]; then gen_ppm_opt="class=1 num_gen=$num_gen"
  elif [ "$type" = "random"       ]; then 
    num_gen=$(( ww*hh ))
    if [ "$numcls" = 2 ]; then gen_ppm_opt="class=0"; fi  
  else
    echo Unknown type: $type
    echo Use random, class0-best, class1-best, each-best, or high-entropy. 
    exit 1
  fi
  genfn=${gendir}/${mynm}-${num_gen}

  echo; echo "*** Generating $num_gen images in the xbin format ... "
  $exe $gpumem cfg_gen Bin \
             model_fn=${moddir}/${modfn} \
             num_gen=$num_gen \
             gen_fn=$genfn \
             mini_batch_size=64 gen_seed=$genseed
  if [ $? != 0 ]; then echo $fnm: generation failed.; exit 1; fi

  #---  Make a collage from generate images in the xbin format. 
  ofn=${genfn}-${type}-${ww}x${hh}.${imgext}
  echo; echo "*** Generating a collage from an xbin file ... "
  $exe $gpumem gen_ppm $gen_ppm_opt w=$ww h=$hh gap=$gap \
           datatype=imgbin tstname=$genfn x_ext=.xbin y_ext=.y Scale-1+1 \
           classif_fn=$cls_fn mini_batch_size=64 \
           gen_fn=$ofn
  if [ $? != 0 ]; then echo $fnm: gen_ppm failed.; exit 1; fi

  echo Done ... output: $ofn 
