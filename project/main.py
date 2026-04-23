import sys


def _missing_dep_message(module_name: str) -> str:
    if module_name == "torch":
        return (
            "Missing dependency: PyTorch (`torch`).\n"
            "Install dependencies first:\n"
            "  python -m pip install -r project/requirements.txt\n"
            "Then rerun:\n"
            "  python -m project.main"
        )
    return (
        f"Missing dependency: `{module_name}`.\n"
        "Install dependencies first:\n"
        "  python -m pip install -r project/requirements.txt"
    )


if __name__ == "__main__":
    try:
        from project.eval.run_experiments import EvalConfig, run_experiments
    except ModuleNotFoundError as exc:
        print(_missing_dep_message(exc.name), file=sys.stderr)
        raise SystemExit(1) from exc

    cfg = EvalConfig(
        bits_list=[2, 3, 4, 6, 8],
        seeds=[0, 1, 2, 3, 4],
        dims=[64, 128, 256],
        token_choices=[32, 64, 128],
    )
    run_experiments(cfg)
