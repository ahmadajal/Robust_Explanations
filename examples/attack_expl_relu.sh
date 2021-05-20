#!/bin/bash
# if [ -d output_expl_relu/add/saliency ]
# then
#   rm -r output_expl_relu/add/saliency
# fi
# mkdir output_expl_relu/add/saliency
# touch output_expl_relu/add/saliency/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add/saliency/ --attack_type 0 0 1 --method "saliency" --lr 0.001 | tee -a output_expl_relu/add/saliency/output.log
# #
# if [ -d output_expl_relu/add_stadv/saliency ]
# then
#   rm -r output_expl_relu/add_stadv/saliency
# fi
# mkdir output_expl_relu/add_stadv/saliency
# touch output_expl_relu/add_stadv/saliency/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_stadv/saliency/ --attack_type 0 1 1 --method "saliency" --lr 0.0004 | tee -a output_expl_relu/add_stadv/saliency/output.log
# #
# if [ -d output_expl_relu/add_recolor/saliency ]
# then
#   rm -r output_expl_relu/add_recolor/saliency
# fi
# mkdir output_expl_relu/add_recolor/saliency
# touch output_expl_relu/add_recolor/saliency/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_recolor/saliency/ --attack_type 1 0 1 --method "saliency" --lr 0.001 | tee -a output_expl_relu/add_recolor/saliency/output.log
# #
# if [ -d output_expl_relu/add_stadv_recolor/saliency ]
# then
#   rm -r output_expl_relu/add_stadv_recolor/saliency
# fi
# mkdir output_expl_relu/add_stadv_recolor/saliency
# touch output_expl_relu/add_stadv_recolor/saliency/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_stadv_recolor/saliency/ --attack_type 1 1 1 --method "saliency" --lr 0.0004 | tee -a output_expl_relu/add_stadv_recolor/saliency/output.log
#################################################
# if [ -d output_expl_relu/add/smooth_grad ]
# then
#   rm -r output_expl_relu/add/smooth_grad
# fi
# mkdir output_expl_relu/add/smooth_grad
# touch output_expl_relu/add/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add/smooth_grad/ --attack_type 0 0 1 --method "saliency" --lr 0.001 --smooth True --additive_lp_bound 0.05 | tee -a output_expl_relu/add/smooth_grad/output.log
# #
# if [ -d output_expl_relu/add_stadv/smooth_grad ]
# then
#   rm -r output_expl_relu/add_stadv/smooth_grad
# fi
# mkdir output_expl_relu/add_stadv/smooth_grad
# touch output_expl_relu/add_stadv/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_stadv/smooth_grad/ --attack_type 0 1 1 --method "saliency" --smooth True --lr 0.0004 | tee -a output_expl_relu/add_stadv/smooth_grad/output.log
# #
# if [ -d output_expl_relu/add_recolor/smooth_grad ]
# then
#   rm -r output_expl_relu/add_recolor/smooth_grad
# fi
# mkdir output_expl_relu/add_recolor/smooth_grad
# touch output_expl_relu/add_recolor/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_recolor/smooth_grad/ --attack_type 1 0 1 --method "saliency" --smooth True --lr 0.0008 | tee -a output_expl_relu/add_recolor/smooth_grad/output.log
#
# if [ -d output_expl_relu/add_stadv_recolor/smooth_grad ]
# then
#   rm -r output_expl_relu/add_stadv_recolor/smooth_grad
# fi
# mkdir output_expl_relu/add_stadv_recolor/smooth_grad
# touch output_expl_relu/add_stadv_recolor/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_stadv_recolor/smooth_grad/ --attack_type 1 1 1 --method "saliency" --smooth True --lr 0.0004 | tee -a output_expl_relu/add_stadv_recolor/smooth_grad/output.log
##########################################################
# if [ -d output_expl_relu/add/uniform_grad ]
# then
#   rm -r output_expl_relu/add/uniform_grad
# fi
# mkdir output_expl_relu/add/uniform_grad
# touch output_expl_relu/add/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add/uniform_grad/ --attack_type 0 0 1 --method "uniform_grad" --lr 0.001 --additive_lp_bound 0.05 | tee -a output_expl_relu/add/uniform_grad/output.log
# #
# if [ -d output_expl_relu/add_stadv/uniform_grad ]
# then
#   rm -r output_expl_relu/add_stadv/uniform_grad
# fi
# mkdir output_expl_relu/add_stadv/uniform_grad
# touch output_expl_relu/add_stadv/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_stadv/uniform_grad/ --attack_type 0 1 1 --method "uniform_grad" --lr 0.0004 | tee -a output_expl_relu/add_stadv/uniform_grad/output.log
# #
# if [ -d output_expl_relu/add_recolor/uniform_grad ]
# then
#   rm -r output_expl_relu/add_recolor/uniform_grad
# fi
# mkdir output_expl_relu/add_recolor/uniform_grad
# touch output_expl_relu/add_recolor/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_recolor/uniform_grad/ --attack_type 1 0 1 --method "uniform_grad" --lr 0.0008 | tee -a output_expl_relu/add_recolor/uniform_grad/output.log
# #
# if [ -d output_expl_relu/add_stadv_recolor/uniform_grad ]
# then
#   rm -r output_expl_relu/add_stadv_recolor/uniform_grad
# fi
# mkdir output_expl_relu/add_stadv_recolor/uniform_grad
# touch output_expl_relu/add_stadv_recolor/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/add_stadv_recolor/uniform_grad/ --attack_type 1 1 1 --method "uniform_grad" --lr 0.0004 | tee -a output_expl_relu/add_stadv_recolor/uniform_grad/output.log
##########################################################ciede###########################
# if [ -d output_expl_relu/ciede/add/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add/smooth_grad
# touch output_expl_relu/ciede/add/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add/smooth_grad/ --attack_type 0 0 1 --method "saliency" --lr 0.001 --smooth True --ciede2000_reg 0.01 --additive_lp_bound 0.06 | tee -a output_expl_relu/ciede/add/smooth_grad/output.log
#
# if [ -d output_expl_relu/ciede/add_stadv/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add_stadv/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add_stadv/smooth_grad
# touch output_expl_relu/ciede/add_stadv/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_stadv/smooth_grad/ --attack_type 0 1 1 --method "saliency" --smooth True --lr 0.0004 --ciede2000_reg 0.01 --additive_lp_bound 0.12 --stadv_lp_bound 0.1 | tee -a output_expl_relu/ciede/add_stadv/smooth_grad/output.log
# #
# if [ -d output_expl_relu/ciede/add_recolor/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add_recolor/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add_recolor/smooth_grad
# touch output_expl_relu/ciede/add_recolor/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_recolor/smooth_grad/ --attack_type 1 0 1 --method "saliency" --smooth True --lr 0.001 --ciede2000_reg 0.01 --additive_lp_bound 0.12 | tee -a output_expl_relu/ciede/add_recolor/smooth_grad/output.log
# #
# if [ -d output_expl_relu/ciede/add_stadv_recolor/smooth_grad ]
# then
#   rm -r output_expl_relu/ciede/add_stadv_recolor/smooth_grad
# fi
# mkdir output_expl_relu/ciede/add_stadv_recolor/smooth_grad
# touch output_expl_relu/ciede/add_stadv_recolor/smooth_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_stadv_recolor/smooth_grad/ --attack_type 1 1 1 --method "saliency" --smooth True --lr 0.0004 --ciede2000_reg 0.01 --additive_lp_bound 0.12 --stadv_lp_bound 0.1 | tee -a output_expl_relu/ciede/add_stadv_recolor/smooth_grad/output.log
##########################################################
# if [ -d output_expl_relu/ciede/add/uniform_grad ]
# then
#   rm -r output_expl_relu/ciede/add/uniform_grad
# fi
# mkdir output_expl_relu/ciede/add/uniform_grad
# touch output_expl_relu/ciede/add/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add/uniform_grad/ --attack_type 0 0 1 --method "uniform_grad" --lr 0.001 --ciede2000_reg 0.01 --additive_lp_bound 0.06 | tee -a output_expl_relu/ciede/add/uniform_grad/output.log
#
# if [ -d output_expl_relu/ciede/add_stadv/uniform_grad ]
# then
#   rm -r output_expl_relu/ciede/add_stadv/uniform_grad
# fi
# mkdir output_expl_relu/ciede/add_stadv/uniform_grad
# touch output_expl_relu/ciede/add_stadv/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_stadv/uniform_grad/ --attack_type 0 1 1 --method "uniform_grad" --lr 0.0004 --ciede2000_reg 0.01 --additive_lp_bound 0.12 --stadv_lp_bound 0.1 | tee -a output_expl_relu/ciede/add_stadv/uniform_grad/output.log
# #
# if [ -d output_expl_relu/ciede/add_recolor/uniform_grad ]
# then
#   rm -r output_expl_relu/ciede/add_recolor/uniform_grad
# fi
# mkdir output_expl_relu/ciede/add_recolor/uniform_grad
# touch output_expl_relu/ciede/add_recolor/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_recolor/uniform_grad/ --attack_type 1 0 1 --method "uniform_grad" --lr 0.001 --ciede2000_reg 0.01 --additive_lp_bound 0.12 | tee -a output_expl_relu/ciede/add_recolor/uniform_grad/output.log
# #
# if [ -d output_expl_relu/ciede/add_stadv_recolor/uniform_grad ]
# then
#   rm -r output_expl_relu/ciede/add_stadv_recolor/uniform_grad
# fi
# mkdir output_expl_relu/ciede/add_stadv_recolor/uniform_grad
# touch output_expl_relu/ciede/add_stadv_recolor/uniform_grad/output.log
# python attack_expl_relu.py --output_dir output_expl_relu/ciede/add_stadv_recolor/uniform_grad/ --attack_type 1 1 1 --method "uniform_grad" --lr 0.0004 --ciede2000_reg 0.01 --additive_lp_bound 0.12 --stadv_lp_bound 0.1 | tee -a output_expl_relu/ciede/add_stadv_recolor/uniform_grad/output.log
###############################################################
