import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as torch_dist

from ignite.engine import Engine
import ignite.distributed as idist

from transforms import extract_params


def prepare_training_batch(batch, t1, t2, device):
    (x, (x1, w1), (x2, w2)), _ = batch
    
    with torch.no_grad():
        x = x.to(device)
        x1 = t1(x1.to(device)).detach()
        x2 = t2(x2.to(device)).detach()
        param1, param2 = extract_params(t1, t2, w1, w2)
        param1 = torch.cat([v.to(device) for v in param1.values()], dim=1)
        param2 = torch.cat([v.to(device) for v in param2.values()], dim=1)
    return x, x1, x2, param1, param2


def simclr(backbone,
           projectors,
           p_projector,
           eq_predictor,
           gatingnetwork,
           t1,
           t2,
           optimizers,
           inv_criterion,
           eq_criterion,
           device,
           dataset
           ):

    def training_step(engine, batch):
        backbone.train()
        projectors.train()
        p_projector.train()
        eq_predictor.train()
        gatingnetwork.train()

        if dataset=='stl10':
            p_mean = torch.tensor([13.0968, 13.1474, 69.8042, 69.7284, 0.4986, 1.0016941e+00, 9.9847072e-01, 9.9962264e-01, -9.0546833e-05, 0.2066, 0.53592795]).to(device)
            p_std = torch.tensor([11.503563, 11.360162, 14.589053, 14.54193, 0.49999804,0.20763062, 0.20504521, 0.20646675, 0.05180895, 0.40486595, 0.65823567]).to(device)
        elif dataset=='imagenet100':
            p_mean = torch.tensor([55.96519, 85.8532, 292.3404, 298.526, 0.50069067, 1.0005505e+00, 1.0010018e+00, 1.0004101e+00, 1.1400196e-04, 0.20029363, 0.5294427]).to(device)
            p_std = torch.tensor([65.29286, 87.91453, 133.99657, 141.05266, 0.49999952, 0.20643362, 0.20659508, 0.20603292, 0.05166076, 0.40022006, 0.65506816]).to(device)
        
        for o in optimizers:
            o.zero_grad()

        x0, x1, x2, param_1, param_2 = prepare_training_batch(batch, t1, t2, device)
        param_1 = (param_1 - p_mean) / p_std
        param_2 = (param_2 - p_mean) / p_std

        y0 = backbone(x0)
        y1 = backbone(x1)
        y2 = backbone(x2)

        w0 = gatingnetwork(y0)  
        w1 = gatingnetwork(y1)
        w2 = gatingnetwork(y2)

        stack0 = torch.stack([project(y0) for project in projectors], dim=-1)
        stack1 = torch.stack([project(y1) for project in projectors], dim=-1)
        stack2 = torch.stack([project(y2) for project in projectors], dim=-1)

        z1 = torch.einsum('boe,be->bo', stack1, w1[0])
        z2 = torch.einsum('boe,be->bo', stack2, w2[0])

        eq0 = torch.einsum('boe,be->bo', stack0, w0[1])
        eq1 = torch.einsum('boe,be->bo', stack1, w1[1])
        eq2 = torch.einsum('boe,be->bo', stack2, w2[1])
        
        p1 = p_projector(param_1)
        p2 = p_projector(param_2)

        eq1_hat = eq0 + eq_predictor(torch.cat([eq0, p1], dim=1))
        eq2_hat = eq0 + eq_predictor(torch.cat([eq0, p2], dim=1))

        eq = torch.cat([eq1, eq2], dim=0)
        eq_hat = torch.cat([eq1_hat, eq2_hat], dim=0)

        loss = inv_criterion(z1, z2)

        outputs = dict(loss=loss, z1=z1, z2=z2)

        eq_loss = eq_criterion(eq, eq_hat)  

        total_loss = loss + eq_loss
        total_loss.backward()

        outputs['eq_loss'] = eq_loss

        for o in optimizers:
            o.step()

        return outputs

    engine = Engine(training_step)
    return engine


def collect_features(backbone,
                     dataloader,
                     device,
                     normalize=True,
                     dst=None,
                     verbose=False):

    if dst is None:
        dst = device

    backbone.eval()
    with torch.no_grad():
        features = []
        labels   = []
        for i, (x, y) in enumerate(dataloader):
            if x.ndim == 5:
                _, n, c, h, w = x.shape
                x = x.view(-1, c, h, w)
                y = y.view(-1, 1).repeat(1, n).view(-1)
            z = backbone(x.to(device))
            if normalize:
                z = F.normalize(z, dim=-1)
            features.append(z.to(dst).detach())
            labels.append(y.to(dst).detach())
            if verbose and (i+1) % 10 == 0:
                print(i+1)
        features = idist.utils.all_gather(torch.cat(features, 0).detach())
        labels   = idist.utils.all_gather(torch.cat(labels, 0).detach())

    return features, labels


def nn_evaluator(backbone,
                 trainloader,
                 testloader,
                 device):

    def evaluator():
        backbone.eval()
        with torch.no_grad():
            features, labels = collect_features(backbone, trainloader, device)
            corrects, total = 0, 0
            for x, y in testloader:
                z = F.normalize(backbone(x.to(device)), dim=-1)
                scores = torch.einsum('ik, jk -> ij', z, features)
                preds = labels[scores.argmax(1)]

                corrects += (preds.cpu() == y).long().sum().item()
                total += y.shape[0]
            corrects = idist.utils.all_reduce(corrects)
            total = idist.utils.all_reduce(total)

        return corrects / total

    return evaluator

