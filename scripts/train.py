import diffuser.utils as utils
import diffuser.models as models
import torch
from tqdm import tqdm
import wandb

def main(**deps):
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
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = False
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    
    wandb.init(project='decdiff-opt',
               config=Config)

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
        # discount=Config.discount,
        # termination_penalty=Config.termination_penalty,
        
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

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
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
            hidden_dim=Config.hidden_dim,
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
        )
    else:
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
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
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
        )

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
        save_checkpoints=Config.save_checkpoints,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    model = model_config()
    proxy_model = proxy_model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    logger.print('✓')

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#
    
    # 1. Training Proxy Model
    trainer.train_proxy(n_train_steps=Config.proxy_n_train_steps)

    # 2. Training Diffusion Model
    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)
    for i in range(n_epochs):
        logger.print(f'Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch)

