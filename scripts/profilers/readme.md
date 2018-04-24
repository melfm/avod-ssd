# TensorFlow Model Profiler

## Requirements
`tensorflow1.3`

## Set up
Make sure you add CUDA libcupti to your path:

`export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH`

---
From the top level avod folder:
```
python scripts/profilers/model_profiler.py
```

This will output number of parameters, shapes and memory information on the console.

---
## Viewing Data
To view the timeline information:
```
Open a Chrome Browser, type URL chrome://tracing, and load the json file.
```
