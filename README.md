# Benchmark for MOT Metrics using Synthetically Degraded Tracking Data

This project serves as a Benchmark for assessing limitations of MOT metrics, using synthetically degraded tracking data. 
It was published with the corresponding paper: [Simulation of Synthetically Degraded Tracking Data to Benchmark MOT Metrics](https://www.researchgate.net/publication/365838978_Simulation_of_Synthetically_Degraded_Tracking_Data_to_Benchmark_MOT_Metrics) on the 32. Workshop Computational Intelligence in Berlin.


## Description
Tracking metrics are considered as an objective measure, thereby we ignore potential
biases and limitations in the tracking metrics themselves.

We propose benchmarking tracking metrics by replacing the tracking algorithm with synthetic tracking results emulating real-world tracking errors. We select a set of frequently occurring real-world tracking errors and provide instructions on how to simulate them.


## Getting Started

### Installation

Follow the steps below to create your local copy. The exact folder structure must be maintained because the code and evaluation methods rely on it.

### 1) create folder structure
The folder structure should look like:
```
PROJECT_NAME
└─── data
└─── synth_tracks_bm
└─── results
```
  
- create a folder e.g. PROJECT_NAME
- change into folder PROJECT_NAME
- create the folders data; results
- change into the folder synth_tracks_bm
- clone the gitlab repository into that folder

          git clone https://github.com/mrhartmann/benchmark-mot-metrics.git


### 2) create virtual environment
Create the virtual environment (either activate conda in cmd window or use the terminal in you prefered IDE, which should usually have conda activated shown by the environment in round brackets before the path 
 (env_name) C:\path\...\)

    conda env create -f env.yaml
    
 activate environment https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#existing-environment
 
 ### 3) add CTC evaluatation metrics
 the metrics are provided as executables and can be downloaded here: 
    
http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip
 
 extract the metrics and move them into a folder called 'CTC_eval'
 so the final folder structure will look like
 
 ```
PROJECT_NAME
└─── data
└─── synth_tracks_bm
     └─── CTC_eval
          └─── Win
          └─── ...
     └─── ...Python Scripts...
└─── results
```

### 4) Get data sets
the data sets Fluo-N2DH-SIM+ and Fluo-N3DH-SIM+ can be downloaded here 

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip

extract zip files to the data folder
 so the final folder structure will look like
 
 ```
PROJECT_NAME
└─── data
    └─── Fluo-N2DH-SIM+
    └─── Fluo-N3DH-SIM+
└─── synth_tracks_bm
     └─── CTC_eval
          └─── Win
          └─── ...
     └─── ...Python Scripts...
└─── results
```
Note: other datasets can be added similarly, however theses usually do not include a ground truth for all data sets (have a look at the 0x_GT SEG and TRA data for that)
But - the silver truth (0x_ST) is often quite good and could be used instead

## Usage
For creating synthetically degraded tracking data, run the "evaluation_script.py" without changing the parameters.
This will create the modified tracking results, evaluate them with the tracking metrics HOTA, MOTA, IDF1 and TRA and finally plot the results.
By commenting out specific tracking errors in the "set_up_synth_track_errors" method, only a subset of tracking errors can be computed. 
By changing the parameters in the "evaluation_script.py", different error percentages or amount of runs can be defined.


This work presents the method and main results of the Bachelor’s thesis by
Hartmann:

_Michael Hartmann. “Simulation of Synthetically Degraded Tracking
Data to Benchmark Multi Object Tracking Metrics”. Bachelor’s thesis.
Karlsruhe: Karlsruhe Institute of Technology (KIT), 2022._


## Acknowledgments

The Project uses functionalities of the following:

* [TrackEval](https://github.com/JonathonLuiten/TrackEval) is used for the evaluation of MOT metrics, including HOTA (developed by Jonathon Luiten and Arne Hoffhues, [HOTA Paper](https://link.springer.com/article/10.1007/s11263-020-01375-2))

* [cocoapi/pycocotools](https://github.com/cocodataset/cocoapi) is used for converting segmentation masks to compressed RLEs and back




