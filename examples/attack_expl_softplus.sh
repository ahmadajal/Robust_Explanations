#!/bin/bash
if [ -d output_expl/add/ ]
then
  rm -r output_expl/add/
fi
mkdir output_expl/add/
touch output_expl/add/output.log
python attack_expl_softplus.py --output_dir output_expl/add/ --attack_type 0 0 1 --num_iter 500 --lr 0.00025 --additive_lp_bound 0.05 | tee -a output_expl/add/output.log
# #
# if [ -d output_expl/add_stadv/ ]
# then
#   rm -r output_expl/add_stadv/
# fi
# mkdir output_expl/add_stadv/
# touch output_expl/add_stadv/output.log
# python attack_expl_softplus.py --output_dir output_expl/add_stadv/ --attack_type 0 1 1 --num_iter 500 --lr 0.0002 | tee -a output_expl/add_stadv/output.log
# #
# if [ -d output_expl/add_recolor/ ]
# then
#   rm -r output_expl/add_recolor/
# fi
# mkdir output_expl/add_recolor/
# touch output_expl/add_recolor/output.log
# python attack_expl_softplus.py --output_dir output_expl/add_recolor/ --attack_type 1 0 1 --num_iter 500 --lr 0.00025 | tee -a output_expl/add_recolor/output.log
# #
# if [ -d output_expl/add_stadv_recolor/ ]
# then
#   rm -r output_expl/add_stadv_recolor/
# fi
# mkdir output_expl/add_stadv_recolor/
# touch output_expl/add_stadv_recolor/output.log
# python attack_expl_softplus.py --output_dir output_expl/add_stadv_recolor/ --attack_type 1 1 1 --num_iter 500 --lr 0.0002 | tee -a output_expl/add_stadv_recolor/output.log
#
if [ -d output_expl/ciede/add ]
then
  rm -r output_expl/ciede/add
fi
mkdir output_expl/ciede/add
touch output_expl/ciede/add/output.log
python attack_expl_softplus.py --output_dir output_expl/ciede/add/ --attack_type 0 0 1 --num_iter 500 --lr 0.00025 --additive_lp_bound 0.06 --ciede2000_reg 0.01 | tee -a output_expl/ciede/add/output.log
#
