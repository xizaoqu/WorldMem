wandb enabled

# set -e
python -m main +name=train \
    dataset.n_frames=8 \
    dataset.save_dir=data/minecraft \
    +dataset.n_frames_valid=700 \
    +dataset.angle_range=110 \
    +dataset.pos_range=8 \
    +dataset.memory_condition_length=8 \
    +dataset.customized_validation=true \
    +dataset.add_timestamp_embedding=true \
    +dataset.wo_updown=false \
    +algorithm.n_tokens=8 \
    +algorithm.memory_condition_length=8 \
    algorithm.context_frames=600 \
    +algorithm.relative_embedding=true \
    +algorithm.log_video=true \
    +algorithm.add_timestamp_embedding=true \
    algorithm.metrics=[lpips,psnr] \
    experiment.training.checkpointing.every_n_train_steps=2500 \
    resume=your_wandb_job_id e.g.yhht29bz \
    +output_dir=your_saving_path e.g. outputs/2025-05-18/15-16-32 \
    experiment.training.max_steps=700000