wandb enabled

# set -e
python -m main +name=train \
    +diffusion_model_path=your_diffusion_model_path \
    +vae_path=your_vae_path \
    +customized_load=true \
    +seperate_load=true \
    +zero_init_gate=true \
    dataset.n_frames=8 \
    dataset.save_dir=data/minecraft \
    +dataset.n_frames_valid=700 \
    +dataset.angle_range=110 \
    +dataset.pos_range=2 \
    +dataset.memory_condition_length=8 \
    +dataset.customized_validation=true \
    +dataset.add_timestamp_embedding=true \
    +dataset.wo_updown=true \
    +algorithm.n_tokens=8 \
    +algorithm.memory_condition_length=8 \
    algorithm.context_frames=600 \
    +algorithm.relative_embedding=true \
    +algorithm.log_video=true \
    +algorithm.add_timestamp_embedding=true \
    algorithm.metrics=[lpips,psnr] \
    experiment.training.checkpointing.every_n_train_steps=2500 \
    experiment.training.max_steps=120000


