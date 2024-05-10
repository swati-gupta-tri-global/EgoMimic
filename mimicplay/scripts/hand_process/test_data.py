import h5py
import numpy as np
import json

issues = {}
for i in range(53):
    try:
        print(f"Testing demo{i}")
        f = h5py.File(f"/coc/flash7/datasets/egoplay/trajPredDatasetv1/demo{i}.h5")
        front_img_1 = f["data/demo_0/obs/front_image_1"]
        count = []
        for j,img in enumerate(front_img_1):
            if not img.any():
                count.append(j)
        issues[f"demo{i}"] = count
    except:
        print("Demo doesn't exist")

with open('corrupted_samples.txt', "w") as file:
    file.write(json.dumps(issues))
