#!/bin/bash
# if [ -d output_expl_relu/add/ ]
# then
#   rm -r output_expl_relu/add/
# fi
# mkdir output_expl_relu/add/
# touch output_expl_relu/add/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add/ --attack_type 0 0 1 | tee -a output_expl_relu/add/output.log
#
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
if [ -d output_expl_relu/add_stadv/ ]
then
  rm -r output_expl_relu/add_stadv/
fi
mkdir output_expl_relu/add_stadv/
touch output_expl_relu/add_stadv/output.log
python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_stadv/ --attack_type 0 1 1 --early_stop_for "expl" --early_stop_value 5.7e-11 | tee -a output_expl_relu/add_stadv/output.log
#
if [ -d output_expl_relu/add_stadv_recolor/ ]
then
  rm -r output_expl_relu/add_stadv_recolor/
fi
mkdir output_expl_relu/add_stadv_recolor/
touch output_expl_relu/add_stadv_recolor/output.log
python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_stadv_recolor/ --attack_type 1 1 1 --early_stop_for "expl" --early_stop_value 5.7e-11 | tee -a output_expl_relu/add_stadv_recolor/output.log
#
# if [ -d output_expl_relu/stadv_recolor/ ]
# then
#   rm -r output_expl_relu/stadv_recolor/
# fi
# mkdir output_expl_relu/stadv_recolor/
# touch output_expl_relu/stadv_recolor/output.log
# python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/stadv_recolor/ --attack_type 1 1 0 | tee -a output_expl_relu/stadv_recolor/output.log
#
if [ -d output_expl_relu/add_recolor/ ]
then
  rm -r output_expl_relu/add_recolor/
fi
mkdir output_expl_relu/add_recolor/
touch output_expl_relu/add_recolor/output.log
python attack_expl_recoloradv_relu.py --output_dir output_expl_relu/add_recolor/ --attack_type 1 0 1 --early_stop_for "expl" --early_stop_value 5.7e-11 | tee -a output_expl_relu/add_recolor/output.log
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
