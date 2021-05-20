#!/bin/bash
# if [ -d output_expl_cifar/standard/add ]
# then
#   rm -r output_expl_cifar/standard/add
# fi
# mkdir output_expl_cifar/standard/add
# touch output_expl_cifar/standard/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add/ --attack_type 0 0 1 --method "saliency" | tee -a output_expl_cifar/standard/add/output.log
#
# if [ -d output_expl_cifar/standard/add_stadv ]
# then
#   rm -r output_expl_cifar/standard/add_stadv
# fi
# mkdir output_expl_cifar/standard/add_stadv
# touch output_expl_cifar/standard/add_stadv/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add_stadv/ --attack_type 0 1 1 --method "saliency" --lr 0.00015 | tee -a output_expl_cifar/standard/add_stadv/output.log
#
# if [ -d output_expl_cifar/standard/add_recolor ]
# then
#   rm -r output_expl_cifar/standard/add_recolor
# fi
# mkdir output_expl_cifar/standard/add_recolor
# touch output_expl_cifar/standard/add_recolor/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add_recolor/ --attack_type 1 0 1 --method "saliency" | tee -a output_expl_cifar/standard/add_recolor/output.log
#
# if [ -d output_expl_cifar/standard/add_stadv_recolor ]
# then
#   rm -r output_expl_cifar/standard/add_stadv_recolor
# fi
# mkdir output_expl_cifar/standard/add_stadv_recolor
# touch output_expl_cifar/standard/add_stadv_recolor/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/standard/add_stadv_recolor/ --attack_type 1 1 1 --method "saliency" --lr 0.00015 | tee -a output_expl_cifar/standard/add_stadv_recolor/output.log
###########################################################
# if [ -d output_expl_cifar/CURE/add ]
# then
#   rm -r output_expl_cifar/CURE/add
# fi
# mkdir output_expl_cifar/CURE/add
# touch output_expl_cifar/CURE/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/CURE/add/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" | tee -a output_expl_cifar/CURE/add/output.log
# #
# if [ -d output_expl_cifar/CURE/add_stadv ]
# then
#   rm -r output_expl_cifar/CURE/add_stadv
# fi
# mkdir output_expl_cifar/CURE/add_stadv
# touch output_expl_cifar/CURE/add_stadv/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/CURE/add_stadv/ --attack_type 0 1 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --lr 0.00015 --additive_lp_bound 0.015 | tee -a output_expl_cifar/CURE/add_stadv/output.log
# #
# if [ -d output_expl_cifar/CURE/add_recolor ]
# then
#   rm -r output_expl_cifar/CURE/add_recolor
# fi
# mkdir output_expl_cifar/CURE/add_recolor
# touch output_expl_cifar/CURE/add_recolor/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/CURE/add_recolor/ --attack_type 1 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --additive_lp_bound 0.015 | tee -a output_expl_cifar/CURE/add_recolor/output.log
# #
# if [ -d output_expl_cifar/CURE/add_stadv_recolor ]
# then
#   rm -r output_expl_cifar/CURE/add_stadv_recolor
# fi
# mkdir output_expl_cifar/CURE/add_stadv_recolor
# touch output_expl_cifar/CURE/add_stadv_recolor/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/CURE/add_stadv_recolor/ --attack_type 1 1 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --lr 0.0001 --additive_lp_bound 0.015 | tee -a output_expl_cifar/CURE/add_stadv_recolor/output.log
#######################################################
if [ -d output_expl_cifar/CURE/add_l1 ]
then
  rm -r output_expl_cifar/CURE/add_l1
fi
mkdir output_expl_cifar/CURE/add_l1
touch output_expl_cifar/CURE/add_l1/output.log
python attack_expl_cifar.py --output_dir output_expl_cifar/CURE/add_l1/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --additive_lp_bound 80.0 | tee -a output_expl_cifar/CURE/add_l1/output.log
#########################################################
# if [ -d output_expl_cifar/standard_RN50/add ]
# then
#   rm -r output_expl_cifar/standard_RN50/add
# fi
# mkdir output_expl_cifar/standard_RN50/add
# touch output_expl_cifar/standard_RN50/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/standard_RN50/add/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN50_standard.pt" | tee -a output_expl_cifar/standard_RN50/add/output.log
#
##########################################################
# if [ -d output_expl_cifar/adv_train_RN50/add ]
# then
#   rm -r output_expl_cifar/adv_train_RN50/add
# fi
# mkdir output_expl_cifar/adv_train_RN50/add
# touch output_expl_cifar/adv_train_RN50/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/adv_train_RN50/add/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --additive_lp_bound 0.05 | tee -a output_expl_cifar/adv_train_RN50/add/output.log
# #
# if [ -d output_expl_cifar/adv_train_RN50/add_stadv ]
# then
#   rm -r output_expl_cifar/adv_train_RN50/add_stadv
# fi
# mkdir output_expl_cifar/adv_train_RN50/add_stadv
# touch output_expl_cifar/adv_train_RN50/add_stadv/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/adv_train_RN50/add_stadv/ --attack_type 0 1 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --lr 0.00015 | tee -a output_expl_cifar/adv_train_RN50/add_stadv/output.log
# #
# if [ -d output_expl_cifar/adv_train_RN50/add_recolor ]
# then
#   rm -r output_expl_cifar/adv_train_RN50/add_recolor
# fi
# mkdir output_expl_cifar/adv_train_RN50/add_recolor
# touch output_expl_cifar/adv_train_RN50/add_recolor/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/adv_train_RN50/add_recolor/ --attack_type 1 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" | tee -a output_expl_cifar/adv_train_RN50/add_recolor/output.log
# #
# if [ -d output_expl_cifar/adv_train_RN50/add_stadv_recolor ]
# then
#   rm -r output_expl_cifar/adv_train_RN50/add_stadv_recolor
# fi
# mkdir output_expl_cifar/adv_train_RN50/add_stadv_recolor
# touch output_expl_cifar/adv_train_RN50/add_stadv_recolor/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/adv_train_RN50/add_stadv_recolor/ --attack_type 1 1 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --lr 0.0001 | tee -a output_expl_cifar/adv_train_RN50/add_stadv_recolor/output.log
#########################################################
if [ -d output_expl_cifar/adv_train_RN50/add_l1 ]
then
  rm -r output_expl_cifar/adv_train_RN50/add_l1
fi
mkdir output_expl_cifar/adv_train_RN50/add_l1
touch output_expl_cifar/adv_train_RN50/add_l1/output.log
python attack_expl_cifar.py --output_dir output_expl_cifar/adv_train_RN50/add_l1/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --additive_lp_bound 120.0 | tee -a output_expl_cifar/adv_train_RN50/add_l1/output.log
##########################CIEDE##########################
# if [ -d output_expl_cifar/ciede/CURE/add ]
# then
#   rm -r output_expl_cifar/ciede/CURE/add
# fi
# mkdir output_expl_cifar/ciede/CURE/add
# touch output_expl_cifar/ciede/CURE/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/ciede/CURE/add/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --additive_lp_bound 0.06 | tee -a output_expl_cifar/ciede/CURE/add/output.log
# #
# if [ -d output_expl_cifar/ciede/CURE/add_reg ]
# then
#   rm -r output_expl_cifar/ciede/CURE/add_reg
# fi
# mkdir output_expl_cifar/ciede/CURE/add_reg
# touch output_expl_cifar/ciede/CURE/add_reg/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/ciede/CURE/add_reg/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --additive_lp_bound 0.06 --ciede2000_reg 0.01 | tee -a output_expl_cifar/ciede/CURE/add_reg/output.log
# #
# if [ -d output_expl_cifar/ciede/adv_train_RN50/add ]
# then
#   rm -r output_expl_cifar/ciede/adv_train_RN50/add
# fi
# mkdir output_expl_cifar/ciede/adv_train_RN50/add
# touch output_expl_cifar/ciede/adv_train_RN50/add/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/ciede/adv_train_RN50/add/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --additive_lp_bound 0.06 | tee -a output_expl_cifar/ciede/adv_train_RN50/add/output.log
# #
# if [ -d output_expl_cifar/ciede/adv_train_RN50/add_reg ]
# then
#   rm -r output_expl_cifar/ciede/adv_train_RN50/add_reg
# fi
# mkdir output_expl_cifar/ciede/adv_train_RN50/add_reg
# touch output_expl_cifar/ciede/adv_train_RN50/add_reg/output.log
# python attack_expl_cifar.py --output_dir output_expl_cifar/ciede/adv_train_RN50/add_reg/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --additive_lp_bound 0.06 --ciede2000_reg 0.01 | tee -a output_expl_cifar/ciede/adv_train_RN50/add_reg/output.log
