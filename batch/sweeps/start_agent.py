import wandb

if __name__ == "__main__":
    wandb.login(key="25ee8d2e5fab3f028de5253bacadfe1ae8bfb760")
    
    wandb.agent(sweep_id="srtea/cee-us/9l4dvui2", count=1)

    #wandb.agent(sweep_id="srtea/cee-us/u7u0y9el", count=1)
    #wandb.agent(sweep_id="srtea/cee-us/msdm6ilz", count=1)
    #wandb.agent(sweep_id="srtea/test/q0hxccgi", count=1)