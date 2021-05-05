#!/bin/bash
# if [ -d output_expl_cifar/standard/add ]
# then
#   rm -r output_expl_cifar/standard/add
# fi
# mkdir output_expl_cifar/standard/add
# touch output_expl_cifar/standard/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add/ --attack_type 0 0 1 --method "saliency" | tee -a output_expl_cifar/standard/add/output.log
#
if [ -d output_expl_cifar/standard/add_stadv ]
then
  rm -r output_expl_cifar/standard/add_stadv
fi
mkdir output_expl_cifar/standard/add_stadv
touch output_expl_cifar/standard/add_stadv/output.log
python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add_stadv/ --attack_type 0 1 1 --method "saliency" | tee -a output_expl_cifar/standard/add_stadv/output.log
#
if [ -d output_expl_cifar/standard/add_recolor ]
then
  rm -r output_expl_cifar/standard/add_recolor
fi
mkdir output_expl_cifar/standard/add_recolor
touch output_expl_cifar/standard/add_recolor/output.log
python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add_recolor/ --attack_type 1 0 1 --method "saliency" | tee -a output_expl_cifar/standard/add_recolor/output.log
#
if [ -d output_expl_cifar/standard/add_stadv_recolor ]
then
  rm -r output_expl_cifar/standard/add_stadv_recolor
fi
mkdir output_expl_cifar/standard/add_stadv_recolor
touch output_expl_cifar/standard/add_stadv_recolor/output.log
python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add_stadv_recolor/ --attack_type 1 1 1 --method "saliency" | tee -a output_expl_cifar/standard/add_stadv_recolor/output.log
