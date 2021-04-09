#!/bin/bash
# if [ -d output_expl/add/ ]
# then
#   rm -r output_expl/add/
# fi
# mkdir output_expl/add/
# touch output_expl/add/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/add/ --norm_weights 0.2 0.2 0.0 --attack_type 0 0 1 --additive_lp_bound 0.08| tee -a output_expl/add/output.log
# #
# if [ -d output_expl/stadv/ ]
# then
#   rm -r output_expl/stadv/
# fi
# mkdir output_expl/stadv/
# touch output_expl/stadv/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/stadv/ --norm_weights 0.2 0.0 0.0 --attack_type 0 1 0 --stadv_lp_bound 0.1| tee -a output_expl/stadv/output.log
# #
# if [ -d output_expl/recolor/ ]
# then
#   rm -r output_expl/recolor/
# fi
# mkdir output_expl/recolor/
# touch output_expl/recolor/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/recolor/ --norm_weights 0.0 0.2 0.0 --attack_type 1 0 0 | tee -a output_expl/recolor/output.log
#
# if [ -d output_expl/add_stadv/ ]
# then
#   rm -r output_expl/add_stadv/
# fi
# mkdir output_expl/add_stadv/
# touch output_expl/add_stadv/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/add_stadv/ --norm_weights 0.2 0.2 0.0 --attack_type 0 1 1 --additive_lp_bound 0.05 --stadv_lp_bound 0.07 --early_stop_for "expl" --early_stop_value 1.6e-11 | tee -a output_expl/add_stadv/output.log
#
# if [ -d output_expl/add_stadv_recolor/ ]
# then
#   rm -r output_expl/add_stadv_recolor/
# fi
# mkdir output_expl/add_stadv_recolor/
# touch output_expl/add_stadv_recolor/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/add_stadv_recolor/ --norm_weights 0.2 0.2 0.0 --attack_type 1 1 1 --additive_lp_bound 0.05 --stadv_lp_bound 0.07 --early_stop_for "expl" --early_stop_value 1.6e-11 | tee -a output_expl/add_stadv_recolor/output.log
# #
# if [ -d output_expl/stadv_recolor/ ]
# then
#   rm -r output_expl/stadv_recolor/
# fi
# mkdir output_expl/stadv_recolor/
# touch output_expl/stadv_recolor/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/stadv_recolor/ --norm_weights 0.1 0.1 0.0 --attack_type 1 1 0 --stadv_lp_bound 0.1 | tee -a output_expl/stadv_recolor/output.log
#
# if [ -d output_expl/add_recolor/ ]
# then
#   rm -r output_expl/add_recolor/
# fi
# mkdir output_expl/add_recolor/
# touch output_expl/add_recolor/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/add_recolor/ --norm_weights 0.2 0.2 0.0 --attack_type 1 0 1 --additive_lp_bound 0.08 --early_stop_for "expl" --early_stop_value 1.6e-11 | tee -a output_expl/add_recolor/output.log
#
# if [ -d output_expl/add_ciede/ ]
# then
#   rm -r output_expl/add_ciede/
# fi
# mkdir output_expl/add_ciede/
# touch output_expl/add_ciede/output.log
# python attack_expl_recoloradv.py --output_dir output_expl/add_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 0 0 1 --additive_lp_bound 0.12 --ciede2000_reg 0.005 | tee -a output_expl/add_ciede/output.log
#
if [ -d output_expl/add_stadv_ciede/ ]
then
  rm -r output_expl/add_stadv_ciede/
fi
mkdir output_expl/add_stadv_ciede/
touch output_expl/add_stadv_ciede/output.log
python attack_expl_recoloradv.py --output_dir output_expl/add_stadv_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 0 1 1 --additive_lp_bound 0.1 --stadv_lp_bound 0.07 --ciede2000_reg 0.005 --early_stop_for "expl" --early_stop_value 1.6e-11 | tee -a output_expl/add_stadv_ciede/output.log
#
if [ -d output_expl/add_recolor_ciede/ ]
then
  rm -r output_expl/add_recolor_ciede/
fi
mkdir output_expl/add_recolor_ciede/
touch output_expl/add_recolor_ciede/output.log
python attack_expl_recoloradv.py --output_dir output_expl/add_recolor_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 1 0 1 --additive_lp_bound 0.12 --ciede2000_reg 0.005 --early_stop_for "expl" --early_stop_value 1.6e-11 | tee -a output_expl/add_recolor_ciede/output.log
#
if [ -d output_expl/add_stadv_recolor_ciede/ ]
then
  rm -r output_expl/add_stadv_recolor_ciede/
fi
mkdir output_expl/add_stadv_recolor_ciede/
touch output_expl/add_stadv_recolor_ciede/output.log
python attack_expl_recoloradv.py --output_dir output_expl/add_stadv_recolor_ciede/ --norm_weights 0.2 0.2 0.0 --attack_type 1 1 1 --additive_lp_bound 0.1 --stadv_lp_bound 0.07 --ciede2000_reg 0.005 --early_stop_for "expl" --early_stop_value 1.6e-11 | tee -a output_expl/add_stadv_recolor_ciede/output.log
