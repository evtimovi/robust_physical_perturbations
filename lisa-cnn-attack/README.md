This directory contains the code for attacking the LISA-CNN model and the model itself. It is self-contained and there should be nothing for you to download in order to run the attack. It also contains the outputs of the runs that generated two of the attacks in the paper (in `optimization_output`).

To run, use the script `run_attack_many.sh`. It is set up in the repo so that it replicates the subliminal poster attack. To see a description of what all the parameters mean, run `python gennoise_many_images.py -h` or look at the definitions of the various command line flags that specify the optimization parameters.

Moreover, the script `run_noise_to_big_img.sh` takes the noise (adversarial perturbation), as stored in some checkpoint file, resizes it, and applies it to a high-res image. 

The "traces" of our optimization runs are stored in these folders:
* `optimization_output/noinversemask_second_trial_run` has the noise and optimization parameters for the subliminal poster attack. The `run_attack_many.sh` is set up to replicate that training run and save it under a folder called `octagon`.

* `optimization_output/l1basedmask_uniformrectangles` contains the outputs from optimizing for a camouflage sticker attack.

In both of these folders, the `model` subfolder contains the final tensorflow checkpoint and `noisy_images` holds images with the perturbation applied to them saved at regular intervals during the attack optimization. 

The `optimization_output_*.txt` files hold the printouts of the optimization parameters. Use these values in `run_noise_to_big_img.sh`  if you want to replicate any one optimization run.