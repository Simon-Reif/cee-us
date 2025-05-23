import wandb

wandb.login(key="25ee8d2e5fab3f028de5253bacadfe1ae8bfb760")
wandb.agent(sweep_id="srtea/cee-us/ccdrxre2", count=1)