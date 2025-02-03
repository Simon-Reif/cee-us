import logging
import os

import torch
from mbrl import allogger, torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.helpers import gen_rollouts
from mbrl.params_utils import read_params_from_cmdline, save_settings_to_json
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl.seeding import Seeding

#self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
#self.logger.log(torch.min(self.costs_).item(), key="best_trajectory_cost")

def eval(controller: ForwardBackwardController, adaptation_data: RolloutBuffer, params, t=None):
    eval_logger = allogger.get_logger(scope="eval", default_outputs=["tensorboard"])
    """"
    num_eval_episodes: 10
    num_inference_samples: 50_000
    eval_tasks: None
    """
    # TODO: maybe inference_batch_size parameter necessary
    if t is not None:
        print(f"Evaluation at iteration {t}")
    obs, actions, next_obs = adaptation_data.get_random_transitions(params.num_inference_samples)
    bs=controller.calculate_Bs(next_obs)
    for task in params.eval_tasks:
        params["env_params"]["case"] = task
        env = env_from_string(params.env, **params.env_params)
        z_r = controller.estimate_z_r(obs, actions, next_obs, env, bs=bs)
        rollout_man = RolloutManager(env, params.rollout_params)
        rollout_buffer = gen_rollouts(
            params,
            rollout_man,
            ,
            initial_controller,
            rollout_buffer,
            forward_model,
            iteration,
            do_initial_rollouts,
        )




def main(params):
    '''
    eval_every_steps: 100_000
    '''
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})

    Seeding.set_seed(params.seed if "seed" in params else None)
    # update params file with seed (either given or generated above)
    params_copy = params._mutable_copy()
    params_copy["seed"] = Seeding.SEED
    save_settings_to_json(params_copy, params.working_dir)
    

    # TODO: load rollout with checkpoint manager
    # TODO: add number of actions to params
    FBController = ForwardBackwardController(params)


    rollout_buffer = RolloutBuffer()  # buffer for main controller/policy rollouts
    buffer = {"rollouts": rollout_buffer}

    checkpoint_manager, _ = maybe_load_checkpoint(
        params=params,
        buffer=buffer,
        imitation=None,
        main_state=main_state,
        main_controller=main_controller,
        forward_model=forward_model,
        reward_info_full=None,
    )

    maybe_prefill_buffer(
        params=params,
        rollout_buffer=rollout_buffer,
    )

    maybe_init_model(
        params=params,
        forward_model=forward_model,
        checkpoint_manager=checkpoint_manager,
        need_pretrained_checkpoint=False,
        env=env,
    )

    maybe_init_controller(
        params=params,
        main_controller=main_controller,
        checkpoint_manager=checkpoint_manager,
        need_pretrained_checkpoint=False,
        env=env,
    )

    do_initial_rollouts = maybe_do_initial_rollouts(
        params=params,
        initial_controller=initial_controller,
        checkpoint_manager=checkpoint_manager,
    )

    total_iterations = params.training_iterations + 1 * do_initial_rollouts
    current_max_iterations = total_iterations

    t_main, gen_main = main_iterator(main_state, current_max_iterations, total_iterations, postfix_dict=None)

    metrics = {}  # Can be updated by any of the hooks

    # --------------------
    # Beginning of Main Loop
    # --------------------

    if "pre_mainloop_hooks" in params:
        hook_executer(params.pre_mainloop_hooks, locals(), globals())

    for iteration in t_main:  # first iteration is for initial controller...

        # --------------------
        # Bookkeeping
        # --------------------

        main_state.iteration = iteration
        is_init_iteration = do_initial_rollouts and iteration == 0

        # --------------------
        # Rollouts
        # --------------------

        if "pre_rollout_hooks" in params:
            hook_executer(params.pre_rollout_hooks, locals(), globals())

        rollout_buffer = gen_rollouts(
            params,
            rollout_man,
            main_controller,
            initial_controller,
            rollout_buffer,
            forward_model,
            iteration,
            do_initial_rollouts,
        )

        if "post_rollout_hooks" in params:
            hook_executer(params.post_rollout_hooks, locals(), globals())

        # --------------------
        # Model learning
        # --------------------

        if "pre_model_learning_hooks" in params:
            hook_executer(params.pre_model_learning_hooks, locals(), globals())

        if getattr(params, "train_model", True):
            train_dynamics_model(
                forward_model,
                rollout_man,
                main_controller,
                buffer,
            )

        if "post_model_learning_hooks" in params:
            hook_executer(params.post_model_learning_hooks, locals(), globals())

        # --------------------
        # Controller learning
        # --------------------

        if "pre_controller_learning_hooks" in params:
            hook_executer(params.pre_controller_learning_hooks, locals(), globals())

        if getattr(params, "train_controller", True):
            train_controller(
                params,
                is_init_iteration,
                main_controller,
                expert_controller,
                rollout_buffer,
                rollout_buffer_expert=None,
            )

        if "post_controller_learning_hooks" in params:
            hook_executer(params.post_controller_learning_hooks, locals(), globals())

        # --------------------
        # Bookkeeping
        # --------------------

        save_checkpoint(
            cpm=checkpoint_manager,
            main_state=main_state,
            buffer=buffer,
            forward_model=forward_model,
            main_controller=main_controller,
            reward_info_full=None,
            final=False,
        )

        allogger.get_root().flush(children=True)

    # --------------------
    # End of Main Loop
    # --------------------

    env.close()
    save_checkpoint(
        cpm=checkpoint_manager,
        main_state=main_state,
        buffer=buffer,
        forward_model=forward_model,
        main_controller=main_controller,
        reward_info_full=None,
        final=True,
    )

    if "post_mainloop_hooks" in params:
        hook_executer(params.post_mainloop_hooks, locals(), globals())

    save_metrics_params(metrics, params)
    print(metrics)

    allogger.close()

    return 0



if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True)

    os.makedirs(params.working_dir, exist_ok=True)

    allogger.basic_configure(
        logdir=params.working_dir,
        default_outputs=["tensorboard"],
        manual_flush=True,
        tensorboard_writer_params=dict(min_time_diff_btw_disc_writes=1),
    )

    allogger.utils.report_env(to_stdout=True)

    save_settings_to_json(params, params.working_dir)

    if "device" in params:
        if "cuda" in params.device:
            if torch.cuda.is_available():
                print(
                    f"Using CUDA device {torch.cuda.current_device()} with compute capability {torch.cuda.get_device_capability(0)}"
                )
                torch_helpers.device = torch.device(params.device)
            else:
                print("CUDA is not available")
        else:
            torch_helpers.device = torch.device(params.device)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    exit(main(params))
