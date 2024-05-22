import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os 
import gym
# from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.utils.des_bench import DesignBenchFunctionWrapper
from diffuser.datasets.d4rl import suppress_output


def evaluate(**deps):
    from ml_logger import logger, RUN

    RUN._update(deps)
    print(deps)
    if deps['task'] == 'ant':
        from config.ant_config import Config
    elif deps['task'] == 'dkitty':
        from config.dkitty_config import Config
    elif deps['task'] == 'tfbind8':
        from config.tfbind8_config import Config
    elif deps['task'] == 'tfbind10':
        from config.tfbind10_config import Config
    elif deps['task'] == 'superconductor':
        from config.superconductor_config import Config
    Config._update(deps)
    
    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'
    
    loadpath = os.path.join(logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{Config.n_train_steps}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)
    
    proxy_loadpath = os.path.join(logger.prefix, 'proxy_checkpoint')
    
    if Config.save_checkpoints:
        proxy_loadpath = os.path.join(proxy_loadpath, f'state_{Config.proxy_n_train_steps}.pt')
    else:
        proxy_loadpath = os.path.join(proxy_loadpath, 'state.pt')
    
    proxy_state_dict = torch.load(proxy_loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        # env=Config.dataset,
        horizon=Config.horizon,
        data_path=Config.data_path,
        context_length=Config.context_length,
        regret=Config.regret,
        # normalizer=Config.normalizer,
        # preprocess_fns=Config.preprocess_fns,
        # use_padding=Config.use_padding,
        # max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        # returns_scale=Config.returns_scale,
    )

    proxy_dataset_config = utils.Config(
        Config.proxy_loader,
        dataset=Config.dataset,
        frac=Config.frac,
        sigma=Config.sigma,
        savepath='proxy_dataset_config.pkl',
    )

    # render_config = utils.Config(
    #     Config.renderer,
    #     savepath='render_config.pkl',
    #     env=Config.dataset,
    # )

    dataset = dataset_config()
    proxy_dataset = proxy_dataset_config()
    # renderer = render_config()
    renderer = Config.renderer
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )
    
    proxy_model_config = utils.Config(
        Config.proxy_model,
        savepath='proxy_model_config.pkl',
        input_dim=observation_dim,
        hidden_dim=Config.proxy_hidden_dim,
        output_dim=action_dim,
        n_ensembles=Config.proxy_n_ensembles,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        # hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )
    
    Config.batch_size = 128
    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        proxy_train_lr=Config.proxy_learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        proxy_save_freq=Config.proxy_save_freq,
        log_freq=Config.log_freq,
        proxy_log_freq=Config.proxy_log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    proxy_model = proxy_model_config()
    diffusion = diffusion_config(model)
    
    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    trainer.proxy_step = proxy_state_dict['step']
    trainer.proxy_model.load_state_dict(proxy_state_dict['model'])
    
    device = Config.device
    context_length = Config.ctx_len
    
    num_queries = 128
    num_eval = 1
    
    contexts = []
    queries = []
    for e in range(num_eval):        
        batch = next(trainer.dataloader)
        
        # context conditioning
        conditions = {i: to_torch(batch.trajectories[:, i+Config.horizon-context_length], device=device) for i in range(context_length)}
        conditions["ctx_len"] = to_torch(np.ones(trainer.batch_size,), device=device) * context_length
        
        # classifier-free guidance
        returns = torch.ones(1, ).to(device=device).unsqueeze(0) * Config.alpha
        returns = returns.repeat(trainer.batch_size, 1)
        
        samples, time = trainer.ema_model.conditional_sample(conditions, values=None, returns=returns)
        samples = samples[..., :observation_dim]
        print(samples.shape)

        queries.append(samples[:, context_length:])
        contexts.append(samples[:, :context_length])

    queries = torch.cat(queries, dim=0).reshape(-1, observation_dim)
    contexts = torch.cat(contexts, dim=0).reshape(-1, observation_dim).cpu().numpy()
    print(queries.shape, contexts.shape)
    
    queries_norm = trainer.proxy_dataset.normalizer.normalize(trainer.dataset.normalizer.unnormalize(queries.cpu())).to(trainer.device)
    queries_proxy_score = trainer.proxy_model(queries_norm).flatten()

    # filtering
    queries = queries[torch.argsort(queries_proxy_score)[-num_queries:]].cpu()
    queries = dataset.normalizer.unnormalize(queries).numpy()
            
    func = DesignBenchFunctionWrapper(deps["task"], normalise=True)
    if deps["task"].startswith("tfbind"):
        queries = func.task.to_integers(queries.reshape(num_queries, -1, 3))
    else:
        queries = queries.reshape(num_queries, -1)
    y = func.task.predict(queries)
    y_norm = (y - func.min) / (func.max - func.min)
    
    logger.print(f"max_ep_reward: {np.max(y)}, median_ep_reward: {np.median(y)}, mean_ep_reward: {np.mean(y)},", color='green')
    logger.log_metrics_summary({f"max_ep_reward": np.max(y), "median_ep_reward": np.median(y), "mean_ep_reward": np.mean(y)})
    
    logger.print(f"nmax_ep_reward: {np.max(y_norm)}, nmedian_ep_reward: {np.median(y_norm)}, nmean_ep_reward: {np.mean(y_norm)},", color='green')
    logger.log_metrics_summary({f"nmax_ep_reward": np.max(y_norm), "nmedian_ep_reward": np.median(y_norm), "nmean_ep_reward": np.mean(y_norm)})
    
    np.savez_compressed(os.path.join(logger.prefix, f'performance_{Config.n_train_steps}_{trainer.batch_size}x{Config.horizon - context_length}_alpha{Config.alpha}'), y=y, y_norm=y_norm, time=time)
    np.savez_compressed(os.path.join(logger.prefix, f'samples_{Config.n_train_steps}_{trainer.batch_size}x{Config.horizon - context_length}_alpha{Config.alpha}'), queries=queries)
