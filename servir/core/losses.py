import numpy as np
import torch 

"""
FSS Surrogate Loss
This function computes an upper bound to the FSS (Fraction Skill Score) 
and outputs the loss value as a torch variable.
Inputs:
    gt: (torch.tensor) ground truth image sequence of shape (batch size, time_instances,  image_length, image_width)
    pred: (torch.tensor) predicted image sequence of shape (batch size, time_instances, image_length, image_width)
    n: (int) value used to construct the averaging kernel (0 <= n < (min(image_length,image_width)/2))
    theta: (float64) threshold used for FSS calculations (for "significant" precipitation)
    c: (float64) small positive constant used to avoid Nan values in loss function
"""
def FSSSurrogateLoss(gt, pred, n, max_precipitation, theta=8, c=1e-6):

    theta = theta /max_precipitation

    # averaging kernel with no padding
    averaging_kernel = torch.nn.AvgPool2d(kernel_size=(2*n+1, 2*n+1), stride=1)
    
    # UF
    psi_gt_u = torch.relu(gt - torch.ones_like(gt)*(theta - 1))
    psi_gt_u = averaging_kernel(psi_gt_u)

    # U\hat{F}
    psi_pred_u = torch.relu(pred - torch.ones_like(pred)*(theta - 1))
    psi_pred_u = averaging_kernel(psi_pred_u)
    
    # LF
    psi_gt_l = torch.relu(torch.ones_like(gt) - torch.relu(- gt + torch.ones_like(gt)*(theta + 1)))
    psi_gt_l = averaging_kernel(psi_gt_l)
    
    # L\hat{F}
    psi_pred_l = torch.relu(torch.ones_like(pred) - torch.relu(- pred + torch.ones_like(pred)*(theta + 1)))
    psi_pred_l = averaging_kernel(psi_pred_l)
    
    # the following terms are used to compute the FSS Surrogate loss as follows
    # FSS_Surrogate = 1 - \frac{1*\sum_{i,j}{LF_{i,j}.L\hat{F}_{i,j}}}{\sum_{i,j}{U\hat{F}_{ij}^2} +\sum_{i,j}{UF_{ij}^2}  + c}
    # FSS_Surrogate = 1- (numerator/denominator) 
    numerator = 2*torch.mul(psi_gt_l,psi_pred_l).sum()
    denominator = torch.square(psi_gt_u).sum() + torch.square(psi_pred_u).sum() + c
    FSS_surrogate = 1- (numerator/denominator)

    return FSS_surrogate



# testing FSS surrogate loss with random inputs 
def main():
    batch_size = 32
    time_instances = 12
    gt = np.random.random((batch_size,time_instances ,80,70))
    gt = torch.from_numpy(gt)
    pred = np.random.random((batch_size,time_instances ,80,70))
    pred = torch.from_numpy(pred)
    FSSSurrogateLoss(gt, pred, n=20)
    
if __name__ == "__main__":
    main()