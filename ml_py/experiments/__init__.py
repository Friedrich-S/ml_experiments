from ..common.experiment import Experiment, ExperimentData


def register_experiment(module, fancy_name):
    class CustomExperiment(Experiment):
        @staticmethod
        def name() -> str:
            return module

        @staticmethod
        def fancy_name() -> str:
            return fancy_name

        def run(self):
            __import__(module, globals=globals(), level=1).run(ExperimentData(self))

    return CustomExperiment


ALL_EXPERIMENTS = [
    register_experiment("diff_vae_seq_model", "Diffusion VAE Sequence Model"),
    register_experiment("graph_arch_tch", "Event-Based Graph Model [PyTorch]"),
    register_experiment("graph_arch", "Event-Based Graph Model"),
    register_experiment("next_input_pred_model", "Next-Input Prediction Model"),
    register_experiment("random_act_test", "Random Activation Test"),
]
