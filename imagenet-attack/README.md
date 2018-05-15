The code in this folder is the most advanced version of the robust physical perturbations (RP2) algorithm described in [Robust Physical-World Attacks on Deep Learning Visual Classification](https://arxiv.org/abs/1707.08945).

# How to run
To run the code:

1. Run `download_inception.sh` to download the InceptionV3 model weights we used when generating attacks.
2. Create a config file in the `config_attack` folder and specify the parameters for your attack. 
* Either of the two files in the folder are good templates (in fact, they are identical). 
* The name of the `.json` file you create in this folder must match the checkpoint file you want to save your attack variables and noise images as. (That's why there are two identical files with different names; if you run `run_attack.py` without any modifications, it won't override the existing attack data we have shared in the repo.)
* For a description of what each parameter is, run `python attack.py -h` or look into `flags.py`.

3. Download the victim and validation images we used from [here](https://drive.google.com/drive/u/1/folders/1DbsJtE6KT3J15TzcCoVrvoHeCVHhSxtc) or create your own sets. Set your `attack_srcdir` and `validation_set` parameters in your config to point to these sets.
* The only requirement is that the number of victim images is a multiple of your validation set and that you set the `attack_batch_size` parameter in your `.json` file accordingly.
* Of course, the images should be `.jpg` or `.png` but resizing is handled in the code.

4. Change the `PREFIX` variable in `run_attack.sh` to have the name of your `.json` file you created above.

5. Run `run_attack.sh`.
* If a folder `output_noisegen/$PREFIX` exists, the script will stop and not execute the attack. See step 4.

# Description of Files and Folders
* `run_attack.py` shell script to save typing and run `attack.py`.
* `run_classify.sh` If you give it a directory as a positional argument, this will classify all `png` and `jpg` images in it with Inception and output the classifications in `classifications.txt` in that same folder.
* `run_extract_noise.sh` extracts the noise from a tensorflow checkpoint, make sure you set the arguments in the file.
* `run_apply_from_image.sh` extracts the noise from an image file, also make sure you set the parameters in the file.
* `attack.py` main attack script. Handles optimization iterations, batching, data loading, saving, accuracy evaluation, loss accounting, etc. Also sets what values the transformations have during optimization (they are randomly generated every batch subject to the parameters specified).
    * Could also be used only to extract a generated perturbation from a Tensorflow checkpoint if `just_extract_noise` is set to true. In that case, the noise stored in `noise_restore_checkpoint` is loaded and applied to images in `apply_folder`. No attack runs in this mode.
* `attack_util.py` contains the code to set up the Tensorflow graph that runs the attack. In theory, this is set up to be fully self-sustained and independent of any Tensorflow sessions and `attack.py`.
* `flags.py` contains all the configuration for the attack that is specified in a config file. The precedence of values for the parameters is default < command line argument < config file. That is anything in the config file overwrites anything specified as a command line arg, which overwrites anything that is given as a default in `flags.py`.
* `output_noisegen` anything the attack generates here is stored in a sub-folder named after the attack identifier
