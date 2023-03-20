import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision


def attention_rollout(As):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """
    
    rollout = As[0]
    for A in As[1:]:
        rollout = torch.matmul(
            0.5*A + 0.5*torch.eye(A.shape[1], device=A.device),
            rollout
        ) # the computation takes care of skip connections
    
    return rollout

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images(imgs, **kwargs):
    """Shows the images provided.
    """
    
    imgs = imgs.clone()

    im = torchvision.utils.make_grid(imgs)

    # show images
    fig = plt.figure(**kwargs)
    imshow(im)

    return fig

def show_attention(imgs, rollout, **kwargs):
    """Shows the images provided with the given attention masks.
    """
    
    imgs = imgs.clone()
    rollout = rollout.clone()

    # normalize between 0 and 1
    for r in rollout:
        r -= r.min()
        r /= r.max()

    # generate a red binary mask
    mask = torch.zeros_like(imgs)
    mask[:,0] = (rollout > 0.2).squeeze()

    # mask images
    alpha = 0.5
    im = torchvision.utils.make_grid(\
        (1 - alpha)*imgs + alpha*((1 - mask)*imgs + mask))

    # show images
    fig = plt.figure(**kwargs)
    imshow(im)

    return fig