import torch
from torchvision.io import read_video
import torchvision
import av

def extract_frames(video_tensor, num_frames):
    total_frames = video_tensor.shape[0]
    
    if num_frames >= total_frames:
        return video_tensor  # Return all frames if requested number is greater or equal to total frames
    
    # Calculate indices for equally spaced frames
    indices = torch.linspace(0, total_frames - 1, num_frames).round().long()
    
    # Extract frames based on the calculated indices
    selected_frames = video_tensor[indices]
    
    return selected_frames


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, num_frames, type, transform=None):
        self.transform = transform
        self.num_frames = num_frames
        self.dataframe = dataframe[dataframe['type'] == type]
        self.dataframe['index_label'] = self.dataframe['label'].astype('category').cat.codes
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        video, audio, info = read_video(self.dataframe['video_path'][idx], pts_unit="sec")
        video = extract_frames(video, self.num_frames)
        if self.transform is not None:
            video = self.transform(video)
        label = self.dataframe['index_label'][idx]
        return video, label