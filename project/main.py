from project.eval.run_experiments import EvalConfig, run_experiments


if __name__ == "__main__":
    cfg = EvalConfig(
        bits_list=[2, 3, 4, 6, 8],
        seeds=[0, 1, 2, 3, 4],
        dims=[64, 128, 256],
        token_choices=[32, 64, 128],
    )
    run_experiments(cfg)
