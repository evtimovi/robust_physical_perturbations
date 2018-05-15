
This repository holds the code (and some results) used in [Robust Physical-World Attacks on Deep Learning Visual Classification](https://arxiv.org/abs/1707.08945). The software carries an [MIT license](https://github.com/evtimovi/robust_physical_perturbations/blob/master/LICENSE).

The folders are as follows:

* `lisa-cnn-attack` holds the code to attack the LISA-CNN that classifies US road signs from the LISA dataset. Contains a model that achieves 91% accuracy on that dataset. This is the most rudimentary implementation of the algorithm.
* `gtsrb-cnn-attack` holds the code that attacks the GTSRB-CNN that classifies German road signs (with the stops replaced with US ones from LISA). Implementation somewhat improved.
* `imagenet-attack` holds the code that attacks the Inception V3 model that operates on ImageNet data. Most advanced implementation of the algorithm.

Further details are given in `README` files in the respective folders. They also specify how to download portions that are needed for the code to run but are not committed here due to size.

Note that in `lisa-cnn-attack` and in `gtsrb-cnn-attack` we include portions of an older version of the [cleverhans](https://github.com/tensorflow/cleverhans) library for compatibility. It carries its own [MIT License](https://github.com/tensorflow/cleverhans/blob/master/LICENSE).