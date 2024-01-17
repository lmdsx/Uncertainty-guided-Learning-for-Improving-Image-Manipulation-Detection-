import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_utils import bce_loss
from torchvision import models

#Predict μ and θ for each pixel in an input image using the ResNet network. The paper utilizes HRNet with dual streams (RGB, SRM) for feature extraction, but it is not the focus of the implementation.
class manipulation_model(nn.Module):
    def __init__(self):
        super(manipulation_model, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18_features = nn.Sequential(*list(resnet18.children())[:-2])
        self.predict_model = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )    
        self.predict_miu = nn.Sequential(
            # nn.Linear(512, 1),
             nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
        )
        self.predict_logtheta2 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            # nn.Linear(512, 1),
            
        )
    def forward(self, x):
        features = self.resnet18_features(x)
        features = self.predict_model(features)
        miu = self.predict_miu(features)
        theta = torch.exp(0.5*self.predict_logtheta2(features))    
        

        return miu,theta,features

#model detail
class UGL_model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = manipulation_model()
        self.get_uncertainty_map = get_uncertainty_map
        self.alpha=0.5
        self.beta=0.5
        self.refine_pred=nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
            # nn.Dropout(0.1),
        )
    def forward(self, x,gt=torch.rand(2,1,256,256)):
        miu,theta,features = self.model(x)
        #get uncertainty map
        uncertainty_map,uncertainty_map_100=get_uncertainty_map(miu,theta)

        miu_bilinear = F.interpolate(miu, size=gt.shape[2:], mode='bilinear', align_corners=False)
        features=F.interpolate(features, size=gt.shape[2:], mode='bilinear', align_corners=False)
        #calculate uncertainty
        uncertainty_loss,uncertainty_map_n,uncertainty_gt = uncertainty_supervision(gt,miu,uncertainty_map,theta)
        
        #refine
        features2=features*(self.alpha*uncertainty_map_n+self.beta*uncertainty_gt)
        
        refine_features=torch.cat([features,features2],dim=1)
        
        refine_image=self.refine_pred(refine_features)
        refine_image=F.interpolate(refine_image, size=gt.shape[2:], mode='bilinear', align_corners=False)
        #bce
        bce_loss_z=bce_loss(refine_image,gt)+bce_loss(miu_bilinear,gt)
        sample_loss=sample_surpervision(gt,uncertainty_map_100)




        return torch.sigmoid(miu_bilinear),torch.sigmoid(refine_image),uncertainty_loss+bce_loss_z+sample_loss
    

    


def get_uncertainty_map(miu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    
    pred = miu + theta * torch.randn_like(torch.rand(100,*miu.shape))
    pred=pred.permute(1, 0, 2, 3,4)
    pred = torch.sigmoid(pred)
    uncertainty_map = pred * torch.log(pred) + (1 - pred) * torch.log(1 - pred)

    return -torch.mean(uncertainty_map, dim=1),pred
     

    

def uncertainty_supervision(gt,coarse_map,uncertainty_map,theta):
    coarse_map = F.interpolate(coarse_map, size=gt.shape[2:], mode='bilinear', align_corners=False)
    theta=F.interpolate(theta, size=gt.shape[2:], mode='bilinear', align_corners=False)
    uncertainty_gt= torch.einsum('bchw,bchw->bchw',gt,(1-coarse_map))+torch.einsum('bchw,bchw->bchw',(1-gt),coarse_map)
  
    # uncertainty_map is normalized to [0, 1]
    uncertainty_map_n = (uncertainty_map-torch.min(uncertainty_map)) / (torch.max(uncertainty_map)-torch.min(uncertainty_map))
    uncertainty_map_n=F.interpolate(uncertainty_map_n,size=gt.shape[2:],mode='bilinear',align_corners=False)
    regularization_term = 0.5 * torch.log(torch.pow(theta,2)+1e-8)

    L2_loss_l=torch.mean(0.5*torch.pow(theta,-2)*((uncertainty_map_n - uncertainty_gt)**2)+regularization_term)
    
    return L2_loss_l,uncertainty_gt,uncertainty_map_n
    
def sample_surpervision(gt,uncertainty_map_100):
    uncertainty_map_100=uncertainty_map_100.squeeze(2)
    uncertainty_map_100=F.interpolate(uncertainty_map_100,size=gt.shape[2:],mode='bilinear',align_corners=False)
    uncertainty_map_100=uncertainty_map_100.unsqueeze(2)
    
    gt=gt.unsqueeze(1).repeat(1,100,1,1,1)
    bce=torch.mean(torch.exp(-gt * torch.log(uncertainty_map_100) - (1 - gt) * torch.log(1 - uncertainty_map_100)),dim=1) #形状：[B,C,H,W]
    sample_loss=torch.mean(torch.log(bce))
    

    
    return sample_loss
    
        
if __name__ == "__main__":
    model=UGL_model(model=manipulation_model)
    image=torch.rand(2,3,256,256)
    gt=torch.rand(2,1,256,256)
    miu,refine_image,loss=model(image,gt)
    print(miu.shape,refine_image.shape,loss)