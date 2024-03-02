import h5py
import os
from mimicplay.scripts.aloha_process.simarUtils import nds

# this is for concatenating together the hand tracking data which comes in separate files.  aloha_to_robomimic is for robot data and will do all processing

def concatenate_h5(source, destination):
    with h5py.File(destination, "w") as dataset:
        for file in os.listdir(source):
            print("trying to add " + file)
            if file.endswith('.h5'):
                try:
                    with h5py.File(os.path.join(source, file), "r") as target:
                        try:
                            demo_group = os.path.splitext(file)[0]
                            demo_num = demo_group.split("demo")[1]
                            demo_group = f"demo_{demo_num}"
                            
                            dataset.create_group(f"data/{demo_group}")
                            # demo_group = dataset.create_group(os.path.splitext(file)[0])
                            target.copy(target["data/demo_0/actions"], dataset[f"data/{demo_group}"])
                            target.copy(target["data/demo_0/obs"], dataset[f"data/{demo_group}"])


                            print("Added " + file)
                        except:
                            print(f"Couldn't add {file}")
                except:
                    print("Couldn't add " + file)
    
    dataset.close()
                

source = "/coc/flash7/datasets/egoplay/handStacking_gray_Public"
destination = "/coc/flash7/datasets/egoplay/handStacking_gray_Public/handStacking_gray.hdf5"

concatenate_h5(source, destination)