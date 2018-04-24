# AVOD-SSD
[1]: https://travis-ci.com/melfm/avod-ssd
[![Build Status](https://travis-ci.com/melfm/avod-ssd.svg?token=EadsqWkUzKDHZjRZYta4&branch=master)][1]

This repository contains the extended version of the original Aggregate View Object Detection ([AVOD](https://github.com/kujason/avod)) network
to run as a 3D Single Stage Detector. This work was done as part of my Master's thesis on [Real-time 3D Object Detection for Autonomous Driving](https://github.com/melfm/masters_thesis).

### KITTI Object Detection Results (3D and BEV)
|              |             |   |           |        AP-3D |           |   |           |       AP-BEV |           |
|:------------:|:-----------:|---|:---------:|:------------:|:---------:|---|:---------:|:------------:|:---------:|
|   **Method** | **Runtime** |   |  **Easy** | **Moderate** |  **Hard** |   |  **Easy** | **Moderate** |  **Hard** |
|        *Car* |             |   |           |              |           |   |           |              |           |
|         AVOD |    **0.08** |   |   73.59   |      65.78   |   58.38   |   |   86.80   |    **85.44** |   77.73   |
|     AVOD-SSD |      0.09   |   |   73.64   |      63.87   |   56.90   |   |   86.14   |      77.66   |   75.68   |
|     AVOD-FPN |      0.10   |   | **81.94** |    **71.88** | **66.38** |   |   88.53   |      83.79   | **77.90** |

Table: Comparison of results of AVOD architectures on the KITTI [3D Object](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and [BEV](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) benchmarks (accessed April 12, 2018).

### KITTI Object Detection Results (3D and BEV)
|              |             |   |      AP-3D    |       AP-BEV |
|:------------:|:-----------:|---|:-------------:|:------------:|
|   **Method** | **Runtime** |   |  **Moderate** | **Moderate** |
|        *Car* |             |   |               |              |
|         AVOD |    **0.08** |   |     74.23     |      73.97   |
|     AVOD-SSD |      0.09   |   |     73.30     |      72.78   |
|     AVOD-FPN |      0.10   |   |    **74.44**  |    **74.11** |

Table: Comparison of results of AVOD architectures on the validation set (50/50 split).

## Getting Started
Implemented and tested on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.3.0.

1. Clone this repo
```bash
git clone git@github.com:melfm/avod-ssd.git --recurse-submodules
```
If you forget to clone the wavedata submodule:
```bash
git submodule update --init --recursive
```

2. Install Python dependencies
```bash
cd avod
pip3 install -r requirements.txt
pip3 install tensorflow-gpu==1.3.0
```

3. Add `avod (top level)` and `wavedata` to your PYTHONPATH
```bash
# For virtualenvwrapper users
add2virtualenv .
add2virtualenv wavedata
```

```bash
# For nonvirtualenv users
export PYTHONPATH=$PYTHONPATH:'/path/to/avod'
export PYTHONPATH=$PYTHONPATH:'/path/to/avod/wavedata'
```

4. Compile integral image library in wavedata
```bash
sh scripts/install/build_integral_image_lib.bash
```

5. Avod uses Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level avod folder):
```bash
sh avod/protos/run_protoc.sh
```

Alternatively, you can run the `protoc` command directly:
```bash
protoc avod/protos/*.proto --python_out=.
```

## Training
### Dataset
To train on the [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
- Download the data and place it in your home folder at `~/Kitti/object`
- Go [here](https://drive.google.com/open?id=1yjCwlSOfAZoPNNqoMtWfEjPCfhRfJB-Z) and download the `train.txt`, `val.txt` and `trainval.txt` splits into `~/Kitti/object`. Also download the `planes` folder into `~/Kitti/object/training`

The folder should look something like the following:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```

### Mini-batch Generation
The training data needs to be pre-processed to generate mini-batches for the RPN(or AVOD-SSD). To configure the mini-batches, you can modify `avod/configs/mb_preprocessing/rpn_[class].config`. You also need to select the *class* you want to train on. Inside the `scripts/preprocessing/gen_mini_batches.py` select the classes to process. By default it processes the *Car* and *People* classes, where the flag `process_[class]` is set to True. The People class includes both Pedestrian and Cyclists. You can also generate mini-batches for a single class such as *Pedestrian* only.

Note: This script does parallel processing with `num_[class]_children` processes for faster processing. This can also be disabled inside the script by setting `in_parallel` to `False`.

```bash
cd avod
python scripts/preprocessing/gen_mini_batches.py
```

Once this script is done, you should now have the following folders inside `avod/data`:
```
data
    label_clusters
    mini_batches
```

### Training Configuration
There are sample configuration files for training inside `avod/configs`. You can train on the example config, or modify an existing configuration. To train a new configuration, copy a config, e.g. `avod_ssd_cars_example.config`, rename this file to a unique experiment name and make sure the file name matches the `checkpoint_name: 'avod_ssd_cars_example'` entry inside your config.

### Run Trainer
To start training, run the following:
```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/avod_ssd_cars_example.config
```
(Optional) Training defaults to using GPU device 1, and the `train` split. You can specify using the GPU device and data split as follows:
```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/avod_ssd_cars_example.config  --device='0' --data_split='train'
```
Depending on your setup, training should take approximately 16 hours with a Titan Xp, and 20 hours with a GTX 1080. If the process was interrupted, training (or evaluation) will continue from the last saved checkpoint if it exists.

### Run Evaluator
To start evaluation, run the following:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/avod_ssd_cars_example.config
```
(Optional) With additional options:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/avod_ssd_cars_example.config --device='0' --data_split='val'
```

The evaluator has two main modes, you can either evaluate a single checkpoint, a list of indices of checkpoints, or repeatedly. The evaluator is designed to be run in parallel with the trainer on the same GPU, to repeatedly evaluate checkpoints. This can be configured inside the same config file (look for `eval_config` entry).

To view the TensorBoard summaries:
```bash
cd avod/data/outputs/avod_ssd_cars_example
tensorboard --logdir logs
```

Note: In addition to evaluating the loss, calculating accuracies, etc, the evaluator also runs the KITTI native evaluation code on each checkpoint. Predictions are converted to KITTI format and the AP is calculated for every checkpoint. The results are saved inside `scripts/offline_eval/results/avod_ssd_cars_example_results_0.1.txt` where `0.1` is the score threshold.

### Run Inference
To run inference on the `val` split, run the following script:
```bash
python avod/experiments/run_inference.py --checkpoint_name='avod_ssd_cars_example' --data_split='val' --ckpt_indices=120 --device='1'
```
The `ckpt_indices` here indicates the indices of the checkpoint in the list. If the `checkpoint_interval` inside your config is `1000`, to evaluate checkpoints `116000` and `120000`, the indices should be `--ckpt_indices=116 120`. You can also just set this to `-1` to evaluate the last checkpoint.

### Viewing Results
All results should be saved in `avod/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/show_predictions_2d.py`. The script needs to be configured to your specific experiments. The `scripts/offline_eval/plot_ap.py` will plot the AP vs. step, and print the 5 highest performing checkpoints for each evaluation metric at the moderate difficulty.

### Model Profiling
You can use the scripts inside `avod/scripts/profilers` to profile the models. The script `model_profiler.py` provides network parameters and FLOP analysis and `inference_speed.py` profiles the run-time speed of various processing stages.

### Suggested Extensions
- Extending the focal loss to the multi-class domain to be able to train on Pedestrian and Cyclists classes.
- Some pre-processing operations could be moved onto the GPU to run faster such as BEV generation.
- AVOD-SSD network runs slightly faster if the final box conversion e.g. `anchors -> box_3d -> box_4c` is cached by being pushed onto the pre-processing stage.

## LICENSE
Copyright (c) 2018 Melissa Mozifian, Jason Ku

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
