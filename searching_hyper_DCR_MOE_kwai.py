import ray
# from ray import tune
# from ray.tune import schedulers
# from ray.tune.schedulers import ASHAScheduler
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from main_function_kwai import run_a_model


def main(num_samples=2, gpus_per_trial=0.5,refer_metrics='val_do_ndcg_post10'):
    config={
        'lr':tune.grid_search([1e-3]),
        'reg_emb':tune.grid_search([1e-6]), #[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,0]
        'reg_para':tune.grid_search([0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]),
        'model':tune.grid_search(['fastFairNFM']), # DCR-MOE
        'alpha':tune.grid_search([0]),
        'stop_refer':tune.grid_search([refer_metrics])
    }
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=refer_metrics,
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["val_condition_ndcg10", "val_do_ndcg10", "val_condition_ndcg_post10", "val_do_ndcg_post10", 'training_iteration'])
    results = tune.run(
        run_a_model,
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="/data/zyang/ray_results") # change to your path

    # scheduler = ASHAScheduler(max_t=100,grace_period=1,reduction_factor=2)
    # results = tune.run(tune.with_parameters(run_a_model),
    #                    resources_per_trial={"cpu": 3, "gpu": gpus_per_trial},
    #                    config=config,
    #                    metric=refer_metrics,
    #                    mode="max",
    #                    num_samples=num_samples,
    #                    scheduler=scheduler)
    best_trial = results.get_best_trial(refer_metrics, "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_do_ndcg10"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result[refer_metrics]))
    print("\n ------------------testing at the best --------------------")
    print("Test the Best Model at the Found Best Trial.....")
    run_a_model(best_trial.config,need_train=False)
    

if __name__=='__main__':
    main(num_samples=1,gpus_per_trial=1,refer_metrics='val_do_ndcg_post10')

