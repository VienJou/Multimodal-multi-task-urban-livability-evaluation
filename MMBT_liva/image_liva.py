"""
This code is adapted from the image.py by Kiela et al. (2020) in https://github.com/facebookresearch/mmbt/blob/master/mmbt/models/image.py
and the equivalent Huggingface implementation: utils_mmimdb.py, which can be
found here: https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/utils_mmimdb.py

The ImageEncoderDenseNet class is modified from the original ImageEncoder class to be based on pre-trained DenseNet
instead of ResNet and to be able to load saved pre-trained weights.

This class makes up the image submodule of the MMBT model.

The forward function is also modified according to the forward function of the DenseNet model listed here:

Original forward function of DenseNet

def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out
"""
import os
import logging
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


logger = logging.getLogger(__name__)

# mapping number of image embeddings to AdaptiveAvgPool2d output size
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

# Default: uses PyTorch's pre-trained DenseNet-121 (ImageNet)
# Optional: can load custom saved model by setting saved_model=True and providing path
MMBT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MMBT_DIR_PARENT, "data_livability")
MODELS_DIR = os.path.join(DATA_DIR, "models")
SAVED_CHEXNET_DEFAULT = os.path.join(MODELS_DIR, "saved_chexnet.pt")

# Find saved_chexnet.pt in multiple possible locations (only used if saved_model=True)
def find_saved_chexnet():
    """Search for saved_chexnet.pt in multiple possible locations (optional, only used if saved_model=True)"""
    possible_paths = [
        # First, check in current project directory
        os.path.join(MMBT_DIR_PARENT, "data_livability", "models", "saved_chexnet.pt"),
        # Check in parent directory's Livability_evaluation subdirectory
        os.path.join(os.path.dirname(MMBT_DIR_PARENT), "Livability_evaluation", "data_livability", "models", "saved_chexnet.pt"),
        # Check in parent directory directly
        os.path.join(os.path.dirname(MMBT_DIR_PARENT), "data_livability", "models", "saved_chexnet.pt"),
    ]
    
    # Also search up to 2 levels in parent directories
    search_path = MMBT_DIR_PARENT
    for _ in range(2):
        possible_paths.append(os.path.join(search_path, "Livability_evaluation", "data_livability", "models", "saved_chexnet.pt"))
        search_path = os.path.dirname(search_path)
        if search_path == os.path.dirname(search_path):  # Reached root
            break
    
    # Find the first existing path
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            logger.info(f"Found saved_chexnet.pt at: {abs_path}")
            return abs_path
    
    # If not found, return default path (may not exist)
    logger.warning(f"saved_chexnet.pt not found in any of the expected locations. Using default: {SAVED_CHEXNET_DEFAULT}")
    return os.path.abspath(SAVED_CHEXNET_DEFAULT)

SAVED_CHEXNET = find_saved_chexnet()


class ImageEncoderDenseNet(nn.Module):
    def __init__(self, num_image_embeds, saved_model=False, path=None):
        """

        :type num_image_embeds: int
        :param num_image_embeds: number of image embeddings to generate; 1-9 as they map to specific numbers of pooling
        output shape in the 'POOLING_BREAKDOWN'
        :param saved_model: True to load saved custom pre-trained model, False to use PyTorch's ImageNet pre-trained DenseNet-121 (default: False)
        :param path: path to the saved .pt model file. If None, will use the automatically found path.
        """
        super().__init__()
        if saved_model:
            # Use provided path or automatically found path
            if path is None:
                path = SAVED_CHEXNET
            
            # Check if file exists
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    f"Please ensure 'saved_chexnet.pt' is in one of the following locations:\n"
                    f"  - {os.path.join(MMBT_DIR_PARENT, 'data_livability', 'models', 'saved_chexnet.pt')}\n"
                    f"  - {os.path.join(os.path.dirname(MMBT_DIR_PARENT), 'Livability_evaluation', 'data_livability', 'models', 'saved_chexnet.pt')}\n"
                    f"Or specify the path explicitly using the 'path' parameter."
                )
            
            # loading custom saved pre-trained weight (e.g., CheXNet or other custom model)
            # the model here expects the weight to be regular Tensors and NOT cuda Tensor
            try:
                model = torch.load(path, weights_only=False)
            except TypeError:
                # Handle older torch versions that do not have weights_only parameter
                model = torch.load(path)
            logger.info(f"Custom saved model loaded from: {path}")
        else:
            # Use PyTorch's pre-trained DenseNet-121 (ImageNet)
            # Try new API first (torchvision >= 0.13), fallback to old API for compatibility
            try:
                model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
            except TypeError:
                # Fallback for older torchvision versions
                model = torchvision.models.densenet121(pretrained=True)
            logger.info("Using PyTorch's ImageNet pre-trained DenseNet-121")

        # DenseNet architecture last layer is the classifier; we only want everything before that
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        # self.model same as original DenseNet self.features part of the forward function
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[3])
       # self.pool = nn.AdaptiveAvgPool2d((3,1))

    def forward(self, input_modal):
        """
        B = batch
        N = number of image embeddings
        1024 DenseNet embedding size, this can be changed when instantiating MMBTconfig for modal_hidden_size

        Bx3x224x224 (this is input shape) -> Bx1024x7x7 (this is shape after DenseNet CNN layers before the last layer)
        -> Bx1024xN (this is after torch.flatten step in this function below) -> BxNx1024 (this is the shape of the
        output tensor)

        :param input_modal: image tensor
        :return:
        """
        # Bx3x224x224 -> Bx1024x7x7 -> Bx1024xN -> BxNx1024
        # 更改之后： Bx9x224x224 -> Bx3x224x224 -> Bx1024x7x7 -> Bx1024xN -> BxNx1024
        #print ("input_modal.size()",input_modal.size()) # 维度
                      
    #    features = self.model(input_modal) # DenseNet 的输出 Bx3x224x224 -> Bx1024x7x7 预训练的网络用的 imagenet？ 是三波段的影像，现在 9个波段能够有用
        
        #print ("tensor_rs",tensor_rs.shape) #tensor_rs torch.Size([15, 3, 224, 224])
        
        tensor_rs, tensor_dsm, tensor_giu = torch.chunk(input_modal,3, dim=1)############# cat chunk  https://blog.csdn.net/Jkwwwwwwwwww/article/details/105726257
        #print ("tensor_rs",tensor_rs.shape) #tensor_rs torch.Size([15, 3, 224, 224])
        features_rs = self.model(tensor_rs)
        features_dsm = self.model(tensor_dsm)
        features_giu = self.model(tensor_giu)
        #print ("features_giu.size",features_giu.size()) #features_giu.size torch.Size([15, 1024, 7, 7])
        
       # features = torch.cat((features_rs,features_dsm,features_giu), dim = 1)
       # print ("features_size_cat",features.size())

        
        out_rs = F.relu(features_rs, inplace=True) 
      #  print ("out1",out_rs.shape) #out1 torch.Size([15, 1024, 7, 7])  
        out_rs = self.pool(out_rs) 
      #  print ("out2",out_rs.shape)  #out2 torch.Size([15, 1024, 3, 1]) 为什么不是([15, 1024, 3, 1])          
        
        out_rs = torch.flatten(out_rs, start_dim=2)
      #  print ("out_3",out_rs.size()) #out_3 torch.Size([15, 1024, 3])  
        out_rs = out_rs.transpose(1, 2).contiguous() # modifies meta information in the Tensor object modifies meta information in the Tensor object
      #  print ("out4",out_rs.shape) #out4 torch.Size([15, 3, 1024])    

        out_dsm = F.relu(features_dsm, inplace=True)
        out_dsm = self.pool(out_dsm)
        out_dsm = torch.flatten(out_dsm, start_dim=2)
        out_dsm = out_dsm.transpose(1, 2).contiguous()
             
        out_giu = F.relu(features_giu, inplace=True)
        out_giu = self.pool(out_giu)
        out_giu = torch.flatten(out_giu, start_dim=2)
        out_giu = out_giu.transpose(1, 2).contiguous()

        out = torch.cat((out_rs,out_dsm,out_giu), dim = 1)
        #print ("out_3data",out.shape) #dim =2； out_3data torch.Size([15, 3, 3072])； dim=1, 15,9,1024


        
        return out  # BxNx1024   
