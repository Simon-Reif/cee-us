import wandb

from mbrl.workflow.inspect_agents import Replay_Manager
from smart_settings.param_classes import recursive_objectify


LAST_ITER = 3_000_000 #checkpoint number

def wandb_summary_success_rates(run, eval_return_dict):
    """
    Update the W&B summary with success rates.
    """
    success_rates = eval_return_dict["success_rates"]
    for task, rate in success_rates.items():
        run.summary[f"success_rate/{task}"] = rate
    print("Updated W&B summary with success rates.")


if __name__ == "__main__":
    wandb.login(key="25ee8d2e5fab3f028de5253bacadfe1ae8bfb760")
    api = wandb.Api()
    # sweep_all = api.sweep("srtea/cee-us/i9rlw389")
    # run_list = sweep_all.runs

    sweep_fp = api.sweep("srtea/cee-us/7dp4cmu2") 
    run_list = list(sweep_fp.runs)
    #run_list = list(sweep_all.runs) + list(sweep_fp.runs)

    for run in run_list:
        #params = recursive_objectify(run.config.as_dict(), make_immutable=False)
        # params = recursive_objectify(run.config, make_immutable=False)
        # working_dir = params.working_dir

        rm = Replay_Manager(working_dir=run.config["working_dir"], iter=LAST_ITER)
        rm.set_wandb_run(run)
        eval_return_dict = rm.eval(num_rollouts=100, log_summary=True)

