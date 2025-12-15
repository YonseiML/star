import random
import torch
import torch.nn as nn
import torch.nn.functional as NF
import torchvision.transforms as T
import torchvision.transforms.functional as F
import kornia
import kornia.augmentation as K
import kornia.augmentation.functional as KF


class MultiView:
    def __init__(self, transform, num_views=2, dataset='stl10'):
        self.transform = transform
        if dataset=='stl10':
            self.no_transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
            ])
        elif dataset=='imagenet100':
            self.no_transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.num_views = num_views

    def __call__(self, x):
        return [self.no_transform(x)] + [self.transform(x) for _ in range(self.num_views)]


class RandomResizedCrop(T.RandomResizedCrop):
    def forward(self, img):
        W, H = F.get_image_size(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        tensor = F.to_tensor(img)
        return tensor, torch.tensor([i, j, h, w], dtype=torch.float)


class ColorJitter(K.ColorJitter):
    def generate_parameters(self, batch_shape: torch.Size):
        params = super().generate_parameters(batch_shape)  
        params['order'] = torch.randperm(4)  
        return params
    def apply_transform(self, x, params):
        transforms = [
            lambda img: KF.apply_adjust_brightness(img, params),
            lambda img: KF.apply_adjust_contrast(img, params),
            lambda img: KF.apply_adjust_saturation(img, params),
            lambda img: KF.apply_adjust_hue(img, params)
        ]

        for idx in params['order'].tolist():
            t = transforms[idx]
            x = t(x)

        return x


class GaussianBlur(K.AugmentationBase2D):
    def __init__(self, kernel_size, sigma, border_type='reflect',
                 return_transform=False, same_on_batch=False, p=0.5):
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.border_type = border_type

    def __repr__(self):
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape):
        return dict(sigma=torch.zeros(batch_shape[0]).uniform_(self.sigma[0], self.sigma[1]))

    def apply_transform(self, input, params):
        sigma = params['sigma'].to(input.device)
        k_half = self.kernel_size // 2
        x = torch.linspace(-k_half, k_half, steps=self.kernel_size, dtype=input.dtype, device=input.device)
        pdf = torch.exp(-0.5*(x[None, :] / sigma[:, None]).pow(2))
        kernel1d = pdf / pdf.sum(1, keepdim=True)
        kernel2d = torch.bmm(kernel1d[:, :, None], kernel1d[:, None, :])
        input = NF.pad(input, (k_half, k_half, k_half, k_half), mode=self.border_type)
        input = NF.conv2d(input.transpose(0, 1), kernel2d[:, None], groups=input.shape[0]).transpose(0, 1)
        return input


def _extract_w(t):
    if isinstance(t, GaussianBlur):
        m = t._params['batch_prob']
        w = torch.zeros(m.shape[0], 1)
        w[m] = t._params['sigma'].unsqueeze(-1)
        return w

    elif isinstance(t, ColorJitter):
        to_apply = t._params['batch_prob']
        w = torch.ones(to_apply.shape[0], 4)
        w[:, 3] = 0  
        
        w[to_apply, 0] = t._params['brightness_factor']
        w[to_apply, 1] = t._params['contrast_factor']
        w[to_apply, 2] = t._params['saturation_factor']
        w[to_apply, 3] = t._params['hue_factor']
        return w

    elif isinstance(t, K.RandomGrayscale):
        to_apply = t._params['batch_prob']
        w = torch.zeros(to_apply.shape[0], 1)
        w[to_apply] = 1 
        return w


def extract_params(transforms1, transforms2, crop1, crop2):
    params1 = {}
    params2 = {}
    for t1, t2 in zip(transforms1, transforms2):
        if isinstance(t1, K.RandomHorizontalFlip):
            f1 = t1._params['batch_prob']
            f2 = t2._params['batch_prob']
            break

    params1['crop'] = crop1
    params2['crop'] = crop2
    params1['flip'] = f1.float().unsqueeze(-1)
    params2['flip'] = f2.float().unsqueeze(-1)

    for t1, t2 in zip(transforms1, transforms2):
        if isinstance(t1, K.RandomHorizontalFlip):
            pass

        elif isinstance(t1, K.ColorJitter):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            params1['color'] = w1
            params2['color'] = w2

        elif isinstance(t1, K.RandomGrayscale):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            params1['grayscale'] = w1 
            params2['grayscale'] = w2

        elif isinstance(t1, GaussianBlur):
            w1 = _extract_w(t1)
            w2 = _extract_w(t2)
            params1['blur'] = w1
            params2['blur'] = w2

        elif isinstance(t1, K.Normalize):
            pass

        elif isinstance(t1, (nn.Identity, nn.Sequential)):
            pass

        else:
            raise Exception(f'Unknown transform: {str(t1.__class__)}')

    return params1, params2
