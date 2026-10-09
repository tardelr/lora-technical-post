from lightning_sdk import Studio, Job, Machine

studio = Studio()

job = Job.run(
    name="train-alpaca",
    command=(
        "cd /teamspace/studios/this_studio/lora-technical-post && "
        "python train_alpaca.py"
    ),
    studio=studio,
    machine=Machine.A100, #overkill for these experiments
)