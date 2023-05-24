# obj-detection-fasterrcnn

## TO GET layer-by-layer profiling results
```
python3 eval.py
```

## TO GET simulated inference time after partitions
```
python3 partition.py
```
Inference time is simulated using the critical path of the model inference.
Critical paths can be generated by the optimizer and are stored in faster-agx/faster-nano folder. (filenames are bandwidth)


## TO GET exported model and model in onnx
```
python3 export.py
```
