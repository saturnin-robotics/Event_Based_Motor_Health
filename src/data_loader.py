import torch
from torch.utils.data import Dataset
import h5py

class EventDataset(Dataset):
    def __init__(self, file_path):
        # Store file path and open the HDF5 file once for fast access
        self.file_path = file_path
        self.data_file = h5py.File(file_path, 'r')
        
        # Calculate the total length of the dataset
        self.data_length = self.data_file['CD']['events'].shape[0]

    def __len__(self):
        # Returns the total number of events
        return self.data_length

    def __getitem__(self, index):
        # Read the structured event record at the specified index
        event = self.data_file['CD']['events'][index]
        
        # Convert the structured record into a simple list of values [x, y, p, t]
        event_values = event.tolist()

        # Convert the list of values into a PyTorch Tensor
        event_tensor = torch.tensor(event_values, dtype=torch.float32) # Ensure float type for ML
        
        # Placeholder label: 0 for 'Healthy' state
        label = 0 
        return event_tensor, label
    
    def __del__(self):
        # Safely close the HDF5 file when the object is destroyed
        if hasattr(self, 'data_file') and self.data_file:
            self.data_file.close()