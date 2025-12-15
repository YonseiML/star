import os
from argparse import ArgumentParser
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite
from ignite.engine import Events
import ignite.distributed as idist

from datasets import load_pretrain_datasets
from models import load_backbone, load_mlp, MultiTaskGatingNetwork
import trainers
from utils import Logger
from objective import *


def simclr(args, t1, t2):
    out_dim = 128
    device = idist.device()

    build_model  = partial(idist.auto_model, sync_bn=True)
    backbone     = build_model(load_backbone(args))
    projectors   = nn.ModuleList([build_model(load_mlp(args.num_backbone_features,
                                        args.num_backbone_features,
                                        out_dim,
                                        num_layers=3,
                                        last_bn=False)) for _ in range(args.expert)])
    p_projector  = build_model(load_mlp(11,
                                        11,
                                        out_dim,
                                        num_layers=1,
                                        last_bn=False))
    eq_predictor = build_model(load_mlp(out_dim * 2,
                                        out_dim * 4,
                                        out_dim,
                                        num_layers=3,
                                        last_bn=False))
    gatingnetwork = build_model(MultiTaskGatingNetwork(args.num_backbone_features, args.expert, 2))

    inv_criterion = NTXent(temperature=args.inv_T, gather_distributed=(args.dataset=='imagenet100'))
    eq_criterion = NTXent(temperature=args.eq_T, gather_distributed=(args.dataset=='imagenet100'))
                                         
    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(backbone.parameters())+\
                              list(projectors.parameters())+\
                              list(p_projector.parameters())+\
                              list(eq_predictor.parameters())+\
                              list(gatingnetwork.parameters()))]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]

    trainer = trainers.simclr(backbone=backbone,
                              projectors=projectors,
                              p_projector=p_projector,
                              eq_predictor=eq_predictor,
                              gatingnetwork=gatingnetwork,
                              t1=t1, t2=t2,
                              optimizers=optimizers,
                              inv_criterion=inv_criterion,
                              eq_criterion=eq_criterion,
                              device=device,
                              dataset=args.dataset)

    return dict(backbone=backbone,
                projectors=projectors,
                p_projector=p_projector,
                eq_predictor=eq_predictor,
                gatingnetwork=gatingnetwork,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()
    logger = Logger(args.logdir, args.resume)

    # DATASETS
    datasets = load_pretrain_datasets(dataset=args.dataset,
                                      datadir=args.datadir)
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=True)
    valloader   = build_dataloader(datasets['val']  , drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)

    t1, t2 = datasets['t1'], datasets['t2']

    # MODELS
    if args.framework == 'simclr':
        models = simclr(args, t1, t2)
    else:
        raise Exception(f'Unknown framework: {args.framework}')

    trainer   = models['trainer']
    evaluator = trainers.nn_evaluator(backbone=models['backbone'],
                                      trainloader=valloader,
                                      testloader=testloader,
                                      device=device)

    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            for loader in [trainloader, valloader, testloader]:
                loader.sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.ITERATION_STARTED)
    def log_lr(engine):
        lrs = {}
        for i, optimizer in enumerate(models['optimizers']):
            for j, pg in enumerate(optimizer.param_groups):
                lrs[f'lr/{i}-{j}'] = pg['lr']
        logger.log(engine, engine.state.iteration, print_msg=False, **lrs)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log(engine):
        loss = engine.state.output.pop('loss')
        eq_loss = engine.state.output.pop('eq_loss')

        logger.log(engine, engine.state.iteration,
                   print_msg=engine.state.iteration % args.print_freq == 0,
                   loss=loss, eq_loss=eq_loss)

        if 'z1' in engine.state.output:
            with torch.no_grad():
                z1 = engine.state.output.pop('z1')
                z2 = engine.state.output.pop('z2')
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                dist = torch.einsum('ik, jk -> ij', z1, z2)
                diag_masks = torch.diag(torch.ones(z1.shape[0])).bool()
                engine.state.output['dist/intra'] = dist[diag_masks].mean().item()
                engine.state.output['dist/inter'] = dist[~diag_masks].mean().item()

    @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_freq))
    def evaluate(engine):
        acc = evaluator()
        logger.log(engine, engine.state.epoch, acc=acc)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        for scheduler in models['schedulers']:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED(every=args.ckpt_freq))
    def save_ckpt(engine):
        logger.save(engine, **models)

    if args.resume is not None:
        @trainer.on(Events.STARTED)
        def load_state(engine):
            ckpt = torch.load(os.path.join(args.logdir, f'ckpt-{args.resume}.pth'), map_location='cpu')
            for k, v in models.items():
                if isinstance(v, nn.parallel.DistributedDataParallel):
                    v = v.module

                if hasattr(v, 'state_dict'):
                    v.load_state_dict(ckpt[k])

                if type(v) is list and hasattr(v[0], 'state_dict'):
                    for i, x in enumerate(v):
                        x.load_state_dict(ckpt[k][i])

    trainer.run(trainloader, max_epochs=args.max_epochs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='stl10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--framework', type=str, default='simclr')

    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--expert', type=int, default=16)
    parser.add_argument('--inv_T', type=float, default=0.2)
    parser.add_argument('--eq_T', type=float, default=0.2)

    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--ckpt-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=1)

    args = parser.parse_args()
    args.lr = args.base_lr * args.batch_size / 256
    if not args.distributed:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        with idist.Parallel('nccl', nproc_per_node=torch.cuda.device_count()) as parallel:
            parallel.run(main, args)

