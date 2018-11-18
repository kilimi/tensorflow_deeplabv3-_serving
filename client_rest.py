import requests
import base64
import json
import matplotlib.pyplot as plt
import numpy as np

image = "some image"
URL = "http://localhost:8501/v1/models/test/versions/1:predict"

headers = {"content-type": "application/json"}
image_content = base64.b64encode(open(image,'rb').read()).decode("utf-8")
body = {
    "signature_name": "predict_image",
    "instances": [
                {"b64":image_content}
    ]
    }
r = requests.post(URL, data=json.dumps(body), headers = headers)

prediction = r.json()['predictions'][0]
im = np.array(prediction).reshape(513, 513)
plt.imshow(im)
plt.show()
