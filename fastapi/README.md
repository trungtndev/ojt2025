docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk


tritonserver --model-repository $(pwd)/model_repository &> server.log
perf_analyzer -m densenet_onnx