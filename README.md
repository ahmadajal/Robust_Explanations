# Robust_Explanations
Code and results for the paper "On the robustness of smoothed explanations"

The structure of the repository is as follows:
- "attacks" folder contains the `mister_ed` framework which was used to to perform explanation attacks.
- "examples" folder contains the scripts to run explanation attacks.
- "notebooks" folder contains the notebooks used to generate the results of the paper.

In order to run explanation attacks, you need to run the scripts in the examples folder with proper attributes. For example, if you want to run a combination of additive and spatial transformation attack against the Gradient explanation of a ReLU network trained on Imagenet you should run the following:
```
python attack_expl_relu.py --output_dir <PATH_TO_OUTPUT_DIR> --attack_type 0 1 1 --method "saliency" --img <PATH_TO_IMAGE> --target_img <PATH_TO_TARGET_IMAGE>
```
