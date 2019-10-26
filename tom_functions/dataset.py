from torch.utils.data import Dataset
from tom_functions.utils import *
from tom_functions.preprocessing import *


class KuzushijiDetectionDatasetTrain(Dataset):
    def __init__(self, data, transform=None):
        self.data      = data[:]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img, heatmap, heightmap, widthmap = preprocessing(self.data[idx])
        h,w,c = heatmap.shape
        targetmap = np.zeros((h,w,3))
        targetmap[:,:,0] = heatmap[:,:,0]
        targetmap[:,:,1] = heightmap[:,:,0]
        targetmap[:,:,2] = widthmap[:,:,0]
        
        if self.transform:
            augmented = self.transform(image=img.astype(np.uint8),
                                       mask=targetmap.astype(np.float32))
            img       = augmented['image']
            targetmap = augmented['mask'][0].transpose(1,2).transpose(0,1) #->(ch,h,w)
            
            #scaling
            targetmap[:,:,:] /= 255.
            
        return {'img':img,'targetmap':targetmap}
    

class KuzushijiDetectionDatasetTest(Dataset):
    def __init__(self, data, transform=None):
        self.data      = data[:]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img = preprocessing(self.data[idx], test_mode=True)
        if self.transform:
            img = self.transform(image=img.astype(np.uint8))['image']
        return {'img':img}