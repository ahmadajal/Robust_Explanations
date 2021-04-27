#!/bin/bash
if [ -d output_expl_relu/add/saliency ]
then
  rm -r output_expl_relu/add/saliency
fi
mkdir output_expl_relu/add/saliency
touch output_expl_relu/add/saliency/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add/saliency/ --attack_type 0 0 1 --method "saliency" --lr 0.001 | tee -a output_expl_relu/add/saliency/output.log

if [ -d output_expl_relu/add_stadv/saliency ]
then
  rm -r output_expl_relu/add_stadv/saliency
fi
mkdir output_expl_relu/add_stadv/saliency
touch output_expl_relu/add_stadv/saliency/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_stadv/saliency/ --attack_type 0 1 1 --method "saliency" --lr 0.0002 | tee -a output_expl_relu/add_stadv/saliency/output.log

if [ -d output_expl_relu/add_recolor/saliency ]
then
  rm -r output_expl_relu/add_recolor/saliency
fi
mkdir output_expl_relu/add_recolor/saliency
touch output_expl_relu/add_recolor/saliency/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_recolor/saliency/ --attack_type 1 0 1 --method "saliency" --lr 0.001 | tee -a output_expl_relu/add_recolor/saliency/output.log

if [ -d output_expl_relu/add_stadv_recolor/saliency ]
then
  rm -r output_expl_relu/add_stadv_recolor/saliency
fi
mkdir output_expl_relu/add_stadv_recolor/saliency
touch output_expl_relu/add_stadv_recolor/saliency/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_stadv_recolor/saliency/ --attack_type 1 1 1 --method "saliency" --lr 0.0002 | tee -a output_expl_relu/add_stadv_recolor/saliency/output.log
#################################################
if [ -d output_expl_relu/add/smooth_grad ]
then
  rm -r output_expl_relu/add/smooth_grad
fi
mkdir output_expl_relu/add/smooth_grad
touch output_expl_relu/add/smooth_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add/smooth_grad/ --attack_type 0 0 1 --method "saliency" --lr 0.001 --smooth True | tee -a output_expl_relu/add/smooth_grad/output.log
#
if [ -d output_expl_relu/add_stadv/smooth_grad ]
then
  rm -r output_expl_relu/add_stadv/smooth_grad
fi
mkdir output_expl_relu/add_stadv/smooth_grad
touch output_expl_relu/add_stadv/smooth_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_stadv/smooth_grad/ --attack_type 0 1 1 --method "saliency" --smooth True --lr 0.0002 | tee -a output_expl_relu/add_stadv/smooth_grad/output.log
#
if [ -d output_expl_relu/add_recolor/smooth_grad ]
then
  rm -r output_expl_relu/add_recolor/smooth_grad
fi
mkdir output_expl_relu/add_recolor/smooth_grad
touch output_expl_relu/add_recolor/smooth_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_recolor/smooth_grad/ --attack_type 1 0 1 --method "saliency" --smooth True --lr 0.001 | tee -a output_expl_relu/add_recolor/smooth_grad/output.log
#
if [ -d output_expl_relu/add_stadv_recolor/smooth_grad ]
then
  rm -r output_expl_relu/add_stadv_recolor/smooth_grad
fi
mkdir output_expl_relu/add_stadv_recolor/smooth_grad
touch output_expl_relu/add_stadv_recolor/smooth_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_stadv_recolor/smooth_grad/ --attack_type 1 1 1 --method "saliency" --smooth True --lr 0.0002 | tee -a output_expl_relu/add_stadv_recolor/smooth_grad/output.log
# ##########################################################
if [ -d output_expl_relu/add/uniform_grad ]
then
  rm -r output_expl_relu/add/uniform_grad
fi
mkdir output_expl_relu/add/uniform_grad
touch output_expl_relu/add/uniform_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add/uniform_grad/ --attack_type 0 0 1 --method "uniform_grad" --lr 0.001 | tee -a output_expl_relu/add/uniform_grad/output.log
#
if [ -d output_expl_relu/add_stadv/uniform_grad ]
then
  rm -r output_expl_relu/add_stadv/uniform_grad
fi
mkdir output_expl_relu/add_stadv/uniform_grad
touch output_expl_relu/add_stadv/uniform_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_stadv/uniform_grad/ --attack_type 0 1 1 --method "uniform_grad" --lr 0.0002 | tee -a output_expl_relu/add_stadv/uniform_grad/output.log
#
if [ -d output_expl_relu/add_recolor/uniform_grad ]
then
  rm -r output_expl_relu/add_recolor/uniform_grad
fi
mkdir output_expl_relu/add_recolor/uniform_grad
touch output_expl_relu/add_recolor/uniform_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_recolor/uniform_grad/ --attack_type 1 0 1 --method "uniform_grad" --lr 0.001 | tee -a output_expl_relu/add_recolor/uniform_grad/output.log
#
if [ -d output_expl_relu/add_stadv_recolor/uniform_grad ]
then
  rm -r output_expl_relu/add_stadv_recolor/uniform_grad
fi
mkdir output_expl_relu/add_stadv_recolor/uniform_grad
touch output_expl_relu/add_stadv_recolor/uniform_grad/output.log
python attack_expl_relu.py --output_dir output_expl_relu/add_stadv_recolor/uniform_grad/ --attack_type 1 1 1 --method "uniform_grad" --lr 0.0002 | tee -a output_expl_relu/add_stadv_recolor/uniform_grad/output.log
##########################################################
# if [ -d output_expl_relu/ciede/add/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add/smooth_grad
# touch output_expl_relu/ciede/add/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add/smooth_grad/ --attack_type 0 0 1 --method "saliency" --lr 0.001 --smooth True --ciede2000_reg 0.0 --additive_lp_bound 0.2 | tee -a output_expl_relu/ciede/add/smooth_grad/output.log
#
# if [ -d output_expl_relu/ciede/add_stadv/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add_stadv/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add_stadv/smooth_grad
# touch output_expl_relu/ciede/add_stadv/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_stadv/smooth_grad/ --attack_type 0 1 1 --method "saliency" --smooth True --lr 0.0002| tee -a output_expl_relu/ciede/add_stadv/smooth_grad/output.log
# #
# if [ -d output_expl_relu/ciede/add_recolor/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add_recolor/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add_recolor/smooth_grad
# touch output_expl_relu/ciede/add_recolor/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_recolor/smooth_grad/ --attack_type 1 0 1 --method "saliency" --smooth True --lr 0.001 | tee -a output_expl_relu/ciede/add_recolor/smooth_grad/output.log
# #
# if [ -d output_expl_relu/ciede/add_stadv_recolor/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add_stadv_recolor/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add_stadv_recolor/smooth_grad
# touch output_expl_relu/ciede/add_stadv_recolor/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_stadv_recolor/smooth_grad/ --attack_type 1 1 1 --method "saliency" --smooth True --lr 0.0002 | tee -a output_expl_relu/ciede/add_stadv_recolor/smooth_grad/output.log
# ##########################################################
# if [ -d output_expl_relu/stadv/ ]
# then
#   rm -r output_expl_relu/stadv/
# fi
# mkdir output_expl_relu/stadv/
# touch output_expl_relu/stadv/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/stadv/ --attack_type 0 1 0 | tee -a output_expl_relu/stadv/output.log
# #
# if [ -d output_expl_relu/recolor/ ]
# then
#   rm -r output_expl_relu/recolor/
# fi
# mkdir output_expl_relu/recolor/
# touch output_expl_relu/recolor/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/recolor/ --attack_type 1 0 0 | tee -a output_expl_relu/recolor/output.log
#
# if [ -d output_expl_relu/add_stadv/ ]
# then
#   rm -r output_expl_relu/add_stadv/
# fi
# mkdir output_expl_relu/add_stadv/
# touch output_expl_relu/add_stadv/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_stadv/ --attack_type 0 1 1 --early_stop_for "expl" --early_stop_value 5.7e-11 | tee -a output_expl_relu/add_stadv/output.log
# #
# if [ -d output_expl_relu/add_stadv_recolor/ ]
# then
#   rm -r output_expl_relu/add_stadv_recolor/
# fi
# mkdir output_expl_relu/add_stadv_recolor/
# touch output_expl_relu/add_stadv_recolor/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_stadv_recolor/ --attack_type 1 1 1 --early_stop_for "expl" --early_stop_value 5.7e-11 | tee -a output_expl_relu/add_stadv_recolor/output.log
#
# if [ -d output_expl_relu/stadv_recolor/ ]
# then
#   rm -r output_expl_relu/stadv_recolor/
# fi
# mkdir output_expl_relu/stadv_recolor/
# touch output_expl_relu/stadv_recolor/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/stadv_recolor/ --attack_type 1 1 0 | tee -a output_expl_relu/stadv_recolor/output.log
#
# if [ -d output_expl_relu/add_recolor/ ]
# then
#   rm -r output_expl_relu/add_recolor/
# fi
# mkdir output_expl_relu/add_recolor/
# touch output_expl_relu/add_recolor/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_recolor/ --attack_type 1 0 1 --early_stop_for "expl" --early_stop_value 5.7e-11 | tee -a output_expl_relu/add_recolor/output.log
#
# if [ -d output_expl_relu/add_ciede/ ]
# then
#   rm -r output_expl_relu/add_ciede/
# fi
# mkdir output_expl_relu/add_ciede/
# touch output_expl_relu/add_ciede/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_ciede/ --attack_type 0 0 1 --ciede2000_reg 0.01 | tee -a output_expl_relu/add_ciede/output.log
#
# if [ -d output_expl_relu/add_stadv_ciede/ ]
# then
#   rm -r output_expl_relu/add_stadv_ciede/
# fi
# mkdir output_expl_relu/add_stadv_ciede/
# touch output_expl_relu/add_stadv_ciede/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_stadv_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 0 1 1 --additive_lp_bound 0.08 --stadv_lp_bound 0.06 --ciede2000_reg 0.01 | tee -a output_expl_relu/add_stadv_ciede/output.log
# #
# if [ -d output_expl_relu/add_recolor_ciede/ ]
# then
#   rm -r output_expl_relu/add_recolor_ciede/
# fi
# mkdir output_expl_relu/add_recolor_ciede/
# touch output_expl_relu/add_recolor_ciede/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_recolor_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 1 0 1 --additive_lp_bound 0.1 --ciede2000_reg 0.01 | tee -a output_expl_relu/add_recolor_ciede/output.log
# #
# if [ -d output_expl_relu/add_stadv_recolor_ciede/ ]
# then
#   rm -r output_expl_relu/add_stadv_recolor_ciede/
# fi
# mkdir output_expl_relu/add_stadv_recolor_ciede/
# touch output_expl_relu/add_stadv_recolor_ciede/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_stadv_recolor_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 1 1 1 --additive_lp_bound 0.08 --stadv_lp_bound 0.06 --ciede2000_reg 0.01 | tee -a output_expl_relu/add_stadv_recolor_ciede/output.log
