from ml_py.experiments import *
from ml_py.common import Experiment
import os


def main():
    experiments = Experiment.__subclasses__()

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )

    print("Available Experiments:")
    for i, e in enumerate(experiments):
        print(f"  {i}: {e.fancy_name()}")

    sel = int(input("Select a number: "))
    if sel < 0 or sel >= len(experiments):
        print(f"{sel} is not a valid experiment number")
        return

    experiment = experiments[sel]
    print(f"Running experiment '{experiment.fancy_name()}'...")
    experiment().run()


if __name__ == "__main__":
    main()
