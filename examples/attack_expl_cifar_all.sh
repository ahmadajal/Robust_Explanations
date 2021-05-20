#!/bin/bash
for i in "59 40" "6 28" "98 2" "68 118" "45 120" "116 36" "118 54" "87 95" "35 74"\
 "117 103" "75 57" "2 37" "67 46" "77 71" "80 42" "28 17" "27 0" "39 85" "9 102"\
  "125 75" "24 66" "22 63" "99 26" "78 10" "89 21"
do
    set -- $i
    echo $1 and $2
    # if [ -d ../ad-hoc_results/CURE/add/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/CURE/add/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/CURE/add/sample_"$1"_"$2"
    # touch ../ad-hoc_results/CURE/add/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/CURE/add//sample_"$1"_"$2"/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/CURE/add/sample_"$1"_"$2"/output.log
    # #
    # if [ -d ../ad-hoc_results/CURE/add_stadv/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/CURE/add_stadv/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/CURE/add_stadv/sample_"$1"_"$2"
    # touch ../ad-hoc_results/CURE/add_stadv/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/CURE/add_stadv/sample_"$1"_"$2"/ --attack_type 0 1 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --lr 0.00015 --additive_lp_bound 0.015 --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/CURE/add_stadv/sample_"$1"_"$2"/output.log
    # #
    # if [ -d ../ad-hoc_results/CURE/add_recolor/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/CURE/add_recolor/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/CURE/add_recolor/sample_"$1"_"$2"
    # touch ../ad-hoc_results/CURE/add_recolor/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/CURE/add_recolor/sample_"$1"_"$2"/ --attack_type 1 0 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --additive_lp_bound 0.015 --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/CURE/add_recolor/sample_"$1"_"$2"/output.log
    # #
    # if [ -d ../ad-hoc_results/CURE/add_stadv_recolor/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/CURE/add_stadv_recolor/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/CURE/add_stadv_recolor/sample_"$1"_"$2"
    # touch ../ad-hoc_results/CURE/add_stadv_recolor/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/CURE/add_stadv_recolor/sample_"$1"_"$2"/ --attack_type 1 1 1 --method "saliency" --model_path "../notebooks/models/RN18_CURE.pth" --lr 0.0001 --additive_lp_bound 0.015 --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/CURE/add_stadv_recolor/sample_"$1"_"$2"/output.log
    ##########################################
    # if [ -d ../ad-hoc_results/adv_train_RN50/add/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/adv_train_RN50/add/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/adv_train_RN50/add/sample_"$1"_"$2"
    # touch ../ad-hoc_results/adv_train_RN50/add/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/adv_train_RN50/add/sample_"$1"_"$2"/ --attack_type 0 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --additive_lp_bound 0.05 --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/adv_train_RN50/add/sample_"$1"_"$2"/output.log
    # #
    # if [ -d ../ad-hoc_results/adv_train_RN50/add_stadv/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/adv_train_RN50/add_stadv/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/adv_train_RN50/add_stadv/sample_"$1"_"$2"
    # touch ../ad-hoc_results/adv_train_RN50/add_stadv/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/adv_train_RN50/add_stadv/sample_"$1"_"$2"/ --attack_type 0 1 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --lr 0.00015 --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/adv_train_RN50/add_stadv/sample_"$1"_"$2"/output.log
    #
    if [ -d ../ad-hoc_results/adv_train_RN50/add_recolor/sample_"$1"_"$2" ]
    then
      rm -r ../ad-hoc_results/adv_train_RN50/add_recolor/sample_"$1"_"$2"
    fi
    mkdir ../ad-hoc_results/adv_train_RN50/add_recolor/sample_"$1"_"$2"
    touch ../ad-hoc_results/adv_train_RN50/add_recolor/sample_"$1"_"$2"/output.log
    python attack_expl_cifar.py --output_dir ../ad-hoc_results/adv_train_RN50/add_recolor/sample_"$1"_"$2"/ --attack_type 1 0 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/adv_train_RN50/add_recolor/sample_"$1"_"$2"/output.log
    #
    # if [ -d ../ad-hoc_results/adv_train_RN50/add_stadv_recolor/sample_"$1"_"$2" ]
    # then
    #   rm -r ../ad-hoc_results/adv_train_RN50/add_stadv_recolor/sample_"$1"_"$2"
    # fi
    # mkdir ../ad-hoc_results/adv_train_RN50/add_stadv_recolor/sample_"$1"_"$2"
    # touch ../ad-hoc_results/adv_train_RN50/add_stadv_recolor/sample_"$1"_"$2"/output.log
    # python attack_expl_cifar.py --output_dir ../ad-hoc_results/adv_train_RN50/add_stadv_recolor/sample_"$1"_"$2"/ --attack_type 1 1 1 --method "saliency" --model_path "../notebooks/models/RN50_linf_8_model_only.pt" --lr 0.0001 --img_idx $1 --target_img_idx $2 | tee -a ../ad-hoc_results/adv_train_RN50/add_stadv_recolor/sample_"$1"_"$2"/output.log
done
