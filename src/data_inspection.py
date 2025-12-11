import h5py
import os

def inspect_dataset(file_path):
    """
    Opens an HDF5 file, lists root keys, and inspects the 'CD' Group content.
    """
    try:
        # Open the file using a context manager for safe closing
        with h5py.File(file_path, 'r') as f:

            print("Keys (Groups/Datasets) found at the file root:")
            
            # Print all top-level keys
            for key in f.keys():
                print(f"  - {key}")

            print("-" * 30)

            # Check if the 'CD' key exists and is a Group
            if 'CD' in f:
                # 'CD' is a Group, not the raw data itself
                cd_group = f['CD'] 
                
                print(f"The 'CD' key is a Group. Listing its contents:")
                
                # Iterate through the sub-elements (which should be the actual Datasets)
                for sub_key in cd_group.keys():
                    cd_data = cd_group[sub_key]
                    
                    # Print the key name, size (Shape), and data format (Dtype)
                    print(f"  Sub-element '{sub_key}':")
                    print(f"    - Size (Shape): {cd_data.shape}")
                    print(f"    - Format (Dtype): {cd_data.dtype}")
                    
    except Exception as e:
        print(f"Error while opening or inspecting the file: {e}")

if __name__ == "__main__":
    
    # Assuming the file is in the 'data/' folder
    file_path = "../data/monitoring_40_50hz.hdf5"
    inspect_dataset(file_path)