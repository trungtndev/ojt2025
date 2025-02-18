### Download triton server image và hosting
Setup project
```
git clone https://github.com/trungtndev/ojt2025
cd host_model
```

```
# Download model if model not exist
wget -O model_repository/densenet_onnx/1/model.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-7.onnx

# Download image for test
wget -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"

```

Run triton server. The triton server auto install if image not exist
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/model
```
Result
```
I0218 18:16:18.769098 1 server.cc:653] 
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| densenet_onnx | 1       | READY  |
+---------------+---------+--------+
...
I0218 18:16:18.770347 1 grpc_server.cc:2450] Started GRPCInferenceService at 0.0.0.0:8001
I0218 18:16:18.770492 1 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
I0218 18:16:18.811449 1 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002
```

### File config for hosting model
host_model/model_repository/densenet_onnx/config.pbtxt
```
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size : 0
input [
  {
    name: "data_0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]
output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    reshape { shape: [ 1, 1000, 1, 1 ] }
    label_filename: "densenet_labels.txt"
  }
]
```
### Call API and run inference
```
 python client.py
```

Result
```
['11.547369:92:BEE EATER' '11.230166:14:INDIGO FINCH'
 '7.527154:95:JACAMAR' '6.921648:17:JAY' '6.576403:88:MACAW']
```
### (Extra - 10đ) Đo performance với toàn bộ config có thể chỉnh sửa từ 2)  - Tutorials Link 3