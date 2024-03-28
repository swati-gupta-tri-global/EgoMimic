import h5py
from simarUtils import *
import numpy as np

# Open the existing HDF5 files
h5py_file1 = h5py.File("/coc/flash7/datasets/egoplay/dino_data/plushiesMimicplay.hdf5", "r")
print(len(h5py_file1['data'].keys()))
h5py_file2 = h5py.File("/coc/flash7/datasets/egoplay/bowl_place_robot_mar4/robomimic/bowl_place_robotMimicplay.hdf5", "r")
print(len(h5py_file2['data'].keys()))
total_len = len(h5py_file1['data'].keys()) + len(h5py_file2["data"].keys())
print('Total len', total_len)

masks = {'train':[], 'valid':[]}

# Create a new HDF5 file to store the combined data
with h5py.File("/coc/flash7/datasets/egoplay/combined_hand_robot_file_dhruv.hdf5", "w") as h5py_combined:

    # Create groups in the combined file
    combined_data = h5py_combined.create_group("data")
    combined_mask = h5py_combined.create_group("mask")

    # Add data and mask from the first file with label 1
    print(h5py_file1['data'].items(), type(h5py_file1['data'].items()))
    for k, v in h5py_file1['data'].items():
        demo_group = combined_data.create_group(k)
        demo_group['label'] = np.array([1])#1  # Add label attribute
        demo_group.attrs['num_samples'] = h5py_file1['data'][k].attrs['num_samples']
        for sub_k, sub_v in v.items():
            if isinstance(sub_v, h5py.Dataset):
                demo_group.create_dataset(sub_k, data=sub_v[()])
            else:  # Handle nested groups
                sub_group = demo_group.create_group(sub_k)
                for sub_sub_k, sub_sub_v in sub_v.items():
                    sub_group.create_dataset(sub_sub_k, data=sub_sub_v[()])


    print("------------------------------")
    print("------------------------------")
    for k, v in h5py_file1['mask'].items():
        # print(k,v)
        # combined_mask.create_dataset(k, data=v[()])

        for demos in h5py_file1['mask'][k][:]:
            masks[k].append(demos)

    print('---------------------------------------------')
    ## Add data and mask from the second file with label 0
    for k, v in h5py_file2['data'].items():
        demo_number = int(k.split('_')[1])
        new_demo_number = demo_number + total_len
        new_demo_name = f'demo_{new_demo_number}'
        demo_group = combined_data.create_group(new_demo_name)
        demo_group['label'] = np.array([0]) #0  # Add label attribute
        demo_group.attrs['num_samples'] = h5py_file2['data'][k].attrs['num_samples']
        for sub_k, sub_v in v.items():
            if isinstance(sub_v, h5py.Dataset):
                demo_group.create_dataset(sub_k, data=sub_v[()])
            else:  # Handle nested groups
                sub_group = demo_group.create_group(sub_k)
                for sub_sub_k, sub_sub_v in sub_v.items():
                    sub_group.create_dataset(sub_sub_k, data=sub_sub_v[()])
    
    print("------------------------------")
    print("------------------------------")
    
    for k, v in h5py_file2['mask'].items():
        print(k, v)
        # combined_mask.create_dataset(k, data=v[()])
        existing_demos = h5py_file2['mask'][k][:]

         # Extract the demo numbers, add 50, and create new demo names
        updated_demos = []
        for demo_name in existing_demos:
            demo_number = int(demo_name.decode().split('_')[1])
            new_demo_number = demo_number + total_len
            updated_demos.append(f'demo_{new_demo_number}'.encode())
        
        masks[k] += updated_demos

        # Convert the list of updated demo names to a NumPy array
        updated_demos_array = np.array(updated_demos, dtype='|S8')
        
        # Write the updated demo names back to the dataset
        existing_demos = updated_demos_array

    for k in masks:
        print(k)
        print(type(masks[k]), masks[k])
        combined_mask.create_dataset(k, data=np.array(masks[k], dtype='|S8'))
    

# Close the files
h5py_file1.close()
h5py_file2.close()
h5py_combined.close()

h5py_combined = h5py.File("/coc/flash7/datasets/egoplay/combined_hand_robot_file_dhruv.hdf5", "r")


nds(h5py_combined)


print('TRAIN', h5py_combined['mask']['train'][:])
print('VALID', h5py_combined['mask']['valid'][:])