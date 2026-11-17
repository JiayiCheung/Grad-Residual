import os

# all outputs (manifests, raw results, checkpoints, claims, analyses) live under one root
RESULTS_ROOT = os.environ.get("GRADRESIDUAL_RESULTS", os.path.join(os.getcwd(), "results"))


def experiment_dir(name):
    return os.path.join(RESULTS_ROOT, name)
