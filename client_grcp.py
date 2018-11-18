from grpc.beta import implementations
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from deeplab import input_preprocess
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import warnings
warnings.filterwarnings("always")

server="localhost:8500"
host, port = server.split(':')
file = 'some image'
im=np.array(Image.open(file))

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "test"
request.model_spec.signature_name = 'predict_image'

filenames = []
filenames.append(file)
files = []
imagedata = []

f = open(file, 'rb')
data = f.read()

#########################################################################################################
start = time.clock()
request.inputs['image'].CopyFrom(make_tensor_proto(data, shape=[1]))
response = stub.Predict(request, 30)
######################################################################################################
output = np.array(response.outputs['seg'].int64_val)
output = np.reshape(output, (513, 513))

######################################################################################################
mask=np.reshape(output, (513, 513))
print ("==>TIME: " , time.clock() - start)

plt.figure(figsize=(14,10))
plt.subplot(1,2,1)
plt.imshow(im, 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(im, 'gray', interpolation='none')
plt.imshow(mask, 'jet', interpolation='none', alpha=0.7)
plt.show()
