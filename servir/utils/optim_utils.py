from torch import optim

optim_parameters = {
    'adam': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
}



def get_optim_scheduler(config, epoch, model, steps_per_epoch):
    opt_lower = config['opt'].lower()

    opt_kwargs = dict.fromkeys(optim_parameters[opt_lower])
    
    for k in opt_kwargs.keys(): 
        opt_kwargs[k] = config[k] if (k in config or f"opt_{k}" in config) and (config[k] is not None)\
              else opt_kwargs.pop(k)

    parameters = model.parameters()

    if opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_kwargs)
    else:
        assert False and "Invalid optimizer"



    # sched_lower = args.sched.lower()
    # total_steps = epoch * steps_per_epoch
    # by_epoch = True
    # if sched_lower == 'onecycle':
    #     lr_scheduler = optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=args.lr,
    #         total_steps=total_steps,
    #         final_div_factor=getattr(args, 'final_div_factor', 1e4))
    #     by_epoch = False
    # elif sched_lower == 'cosine':
    #     lr_scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=epoch,
    #         lr_min=args.min_lr,
    #         warmup_lr_init=args.warmup_lr,
    #         warmup_t=args.warmup_epoch,
    #         t_in_epochs=True,  # update lr by_epoch
    #         k_decay=getattr(args, 'lr_k_decay', 1.0))
    # elif sched_lower == 'tanh':
    #     lr_scheduler = TanhLRScheduler(
    #         optimizer,
    #         t_initial=epoch,
    #         lr_min=args.min_lr,
    #         warmup_lr_init=args.warmup_lr,
    #         warmup_t=args.warmup_epoch,
    #         t_in_epochs=True)  # update lr by_epoch
    # elif sched_lower == 'step':
    #     lr_scheduler = StepLRScheduler(
    #         optimizer,
    #         decay_t=args.decay_epoch,
    #         decay_rate=args.decay_rate,
    #         warmup_lr_init=args.warmup_lr,
    #         warmup_t=args.warmup_epoch)
    # elif sched_lower == 'multistep':
    #     lr_scheduler = MultiStepLRScheduler(
    #         optimizer,
    #         decay_t=args.decay_epoch,
    #         decay_rate=args.decay_rate,
    #         warmup_lr_init=args.warmup_lr,
    #         warmup_t=args.warmup_epoch)
    # else:
    #     assert False and "Invalid scheduler"

    return optimizer, lr_scheduler, by_epoch