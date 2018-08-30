This directory contains the code for attacking the LISA-CNN model and the model itself. It is self-contained and there should be nothing for you to download in order to run the attack. It also contains the outputs of the runs that generated two of the attacks in the paper (in `optimization_output`).

## Important: Tensorflow and Keras Versions
This code runs with tensorflow version 1.4.1 and keras version 1.2.0. We recommend following these steps to ensure that you're not running into version mismatch problems:

1. Make sure you have [pipenv](https://docs.pipenv.org/) installed.
2. From the top-level folder of the repo, execute the following commands:
```
cd lisa-cnn-attack
rm Pipfile
pipenv install tensorflow==1.4.1 
```
(change this to `pipenv install tensorflow-gpu==1.4.1`, if you have a GPU on your system)

```
pipenv install keras==1.2.0
pipenv install scipy
pipenv install opencv-python
pipenv install pillow
pipenv shell
```

3. The last command opens up a pipenv shell for you and in it, `run_attack_many.sh` should run fine. 

## Driver Scripts

To run, use the script `run_attack_many.sh` inside a [pipenv](https://docs.pipenv.org/) shell. It is set up in the repo so that it replicates the subliminal poster attack. To see a description of what all the parameters mean, run `python gennoise_many_images.py -h` or look at the definitions of the various command line flags that specify the optimization parameters. 

The file `Pipfile` specifies the exact version of the packages we used. Newer versions of tensorflow and keras don't always work  with this code. We also include an older version of cleverhans (see below).

Moreover, the script `run_noise_to_big_img.sh` takes the noise (adversarial perturbation), as stored in some checkpoint file, resizes it, and applies it to a high-res image. 

## Outputs of Optimization Runs
The "traces" of our optimization runs are stored in these folders:
* `optimization_output/noinversemask_second_trial_run` has the noise and optimization parameters for the subliminal poster attack. The `run_attack_many.sh` is set up to replicate that training run and save it under a folder called `octagon`.

* `optimization_output/l1basedmask_uniformrectangles` contains the outputs from optimizing for a camouflage sticker attack.

In both of these folders, the `model` subfolder contains the final tensorflow checkpoint and `noisy_images` holds images with the perturbation applied to them saved at regular intervals during the attack optimization. 

The `optimization_output_*.txt` files hold the printouts of the optimization parameters. Use these values in `run_noise_to_big_img.sh`  if you want to replicate any one optimization run.

## Classify Using the Model
The model is to be found under `models/all_r_ivan`. To classify images using it run `python manyclassify.py --attack_srcdir <folder>` where `<folder>` is the path to a folder **of only 32 by 32 png images**. This code is *not* set up to auto-resize images or throw away non-png files in the directory, so it might error out if you don't follow that guideline.

## Attack Code and Cleverhans
The attack graph itself and the code to run it are in the files `gennoise_many_images.py` and `utils.py`. We also include an older version of the core of the [cleverhans](https://github.com/tensorflow/cleverhans) library. It carries its own MIT license.

