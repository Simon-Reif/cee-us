import wandb

if __name__ == "__main__":
    wandb.login(key="25ee8d2e5fab3f028de5253bacadfe1ae8bfb760")

    wandb.agent(sweep_id="srtea/cee-us/n6a765ua", count=1) # sweep12: weight vs max eps zdim50

    
    #wandb.agent(sweep_id="srtea/cee-us/zmyxxwmf", count=1) # sweep6: fp data ablation 

    #wandb.agent(sweep_id="srtea/cee-us/d6z8bjvr", count=1) # sweep6: exp data ablation 1
    #wandb.agent(sweep_id="srtea/cee-us/7dp4cmu2", count=1) # sweep5: final params fg_only
    #wandb.agent(sweep_id="srtea/cee-us/i9rlw389", count=1) # sweep4: final params all
    #wandb.agent(sweep_id="srtea/cee-us/8d60ut7f", count=1)
    #wandb.agent(sweep_id="srtea/cee-us/u7u0y9el", count=1)
    #wandb.agent(sweep_id="srtea/cee-us/msdm6ilz", count=1)
    #wandb.agent(sweep_id="srtea/test/q0hxccgi", count=1)