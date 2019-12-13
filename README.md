# EvalDNN

EvalDNN is an open-source toolbox for model evaluation of deep learning systems, supporting multiple frameworks and metrics.

Author: Yongqiang Tian*, Zhihua Zeng*, Ming Wen, Yepang Liu, Tzu-yang Kuo,
and Shing-Chi, Cheung.

\*The first two author contribute equally. 

This project is mainly supported by $Microsoft Asia Cloud Research Software Fellow Award 2019$

A video is here: the link to youtube

A paper to inroduce this tool is in submit and will be released soon. 

## Frameworks and Metrics
EvalDNN supports the model based on following frameworks:

- TensorFlow
- PyTorch
- Keras
- MXNet

EvalDNN supports the model based on following metrics:

- Top-K accuracy
- Neuron Coverage
- Robustness

## Usage

### Installation
```
pip install EvalDNN
```
### Evaluate a model

Check `demo/demo.ipynb`. 

More examples are avaiable to the `evaldnn/benchmarks/` and `evaldnn/tests`
The examples covers all 4 frameworks and 3 metrics.

### Extension

#### Add a new framework
Create a new .py under `evaldnn.models` then follow the exising implementation in `evaldnn.models`

#### Add a new metric
Create a new .py under `evaldnn.metrics` then follow the exising implementation in `evaldnn.metrics`

## Benchmark

The full benchmark is available here: https://yqtianust.github.io/EvalDNN-benchmark/index.html

The code to reproduce the results in benchmark is in `evaldnn/benchmarks/`.
For example, run 
```
python3 evaldnn/benchmarks/eval_keras
```