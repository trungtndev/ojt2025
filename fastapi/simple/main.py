from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import io
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
import numpy as np

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transformed_img = transform(image).numpy()
    return transformed_img
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    input_tensor = preprocess_image(image)

    client = httpclient.InferenceServerClient(url="localhost:8000")

    inputs = httpclient.InferInput("data_0", input_tensor.shape, datatype="FP32")
    inputs.set_data_from_numpy(input_tensor, binary_data=True)

    outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True, class_count=1000)

    # Querying the server
    results = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("fc6_1").astype(str)

    re = np.squeeze(inference_output)[1]
    return {
        "label": re
    }

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = await file.read()
#     input_tensor = preprocess_image(image)
#
#     payload = {
#         "inputs": [
#             {
#                 "name": "data_0",
#                 "shape": list(input_tensor.shape),
#                 "datatype": "FP32",
#                 "data": input_tensor.flatten().tolist()
#             }
#         ]
#     }
#
#     response = requests.post("http://localhost:8000/v2/models/densenet_onnx/infer", json=payload)
#     result = response.json()
#     return {"predictions": result}


