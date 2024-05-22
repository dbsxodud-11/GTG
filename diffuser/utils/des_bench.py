import sys
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    # from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    # from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
    # from design_bench.datasets.discrete.utr_dataset import UTRDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset

import numpy as np
import torch
import pickle
from botorch.test_functions import Branin


from diffuser.utils.forward import ForwardModel, ProbabilisticForwardModel

TASKNAME2TASK = {
        'dkitty': 'DKittyMorphology-Exact-v0',
        'ant': 'AntMorphology-Exact-v0',
        'tfbind8': 'TFBind8-Exact-v0',
        'tfbind10': 'TFBind10-Exact-v0',
        'superconductor': 'Superconductor-RandomForest-v0',
        'hopper': 'HopperController-Exact-v0',
        'utr': 'UTR-ResNet-v0',
        # 'nas': 'CIFARNAS-Exact-v0',
        # 'chembl': 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0',
        }

class BraninTask:

    def __init__(self, path="generated_datasets/branin/branin_unif_4500.p"):
        data = pickle.load(open(path, "rb"))
        self.x = data[0].astype(np.float32)
        self.y = -data[1].astype(np.float32)

        self.mean_x = self.x.mean(axis=0)
        self.std_x = self.x.std(axis=0)

        self.mean_y = self.y.mean(axis=0)
        self.std_y = self.y.std(axis=0)

        self.is_x_normalized = False
        self.is_y_normalized = False

        self.is_discrete = False
        self.obj_func = Branin(negate=True)


    def map_normalize_x(self):
        self.x = (self.x - self.mean_x) / self.std_x
        self.is_x_normalized = True

    def map_normalize_y(self):
        self.y = (self.y - self.mean_y) / self.std_y
        self.is_y_normalized = True

    def predict(self, x):
        if self.is_x_normalized:
            x = x * self.std_x + self.mean_x
        x[:, 0] = np.clip(x[:, 0], self.obj_func.bounds[0, 0], self.obj_func.bounds[1, 0])
        x[:, 1] = np.clip(x[:, 1], self.obj_func.bounds[0, 1], self.obj_func.bounds[1, 1])

        return self.obj_func(torch.from_numpy(x)).cpu().numpy()

    def denormalize_y(self, y):
        return y * self.std_y + self.mean_y

class DesignBenchFunctionWrapper:
    def __init__(self, taskname, normalise=False, optima=1, oracle=True):
        self.optima = optima
        self.taskname = taskname
        if self.taskname == "branin":
            self.task = BraninTask()
        else:
            self.task = design_bench.make(TASKNAME2TASK[self.taskname])

        self.oracle = oracle
        if (not self.oracle):
            self.forward_net = ForwardModel(hidden_size=128, input_size=self.task.x.shape[-1])
            # self.forward_net = ProbabilisticForwardModel(hidden_size=128, input_size=self.task.x.shape[-1])
            self.forward_net = torch.nn.DataParallel(self.forward_net).to('cuda')
            self.forward_net.load_state_dict(torch.load(f"forward_checkpoints/{taskname}_best"))
            # self.forward_net.load_state_dict(torch.load(f"forward_checkpoints/probabilistic_{taskname}_best"))

        self.max = None
        self.min = None
        self.normalise = normalise
        if (normalise):
            # override optima to be 1 if normalised
            self.optima = 1
            # self.task.map_normalize_y()
            fully_observed_task = None
            if self.taskname == 'tfbind8':
                fully_observed_task = TFBind8Dataset()
            elif self.taskname == 'tfbind10':
                fully_observed_task = TFBind10Dataset()
            elif self.taskname == 'dkitty':
                fully_observed_task = DKittyMorphologyDataset()
            elif self.taskname == 'ant':
                fully_observed_task = AntMorphologyDataset()
            elif self.taskname == 'superconductor':
                fully_observed_task = SuperconductorDataset()
            elif self.taskname == 'hopper':
                fully_observed_task = HopperControllerDataset()
            # elif self.taskname == 'utr':
            #     fully_observed_task = UTRDataset()
            # elif self.taskname == 'nas':
            #     fully_observed_task = CIFARNASDataset()
            # elif self.taskname == 'chembl':
            #     assay_chembl_id = 'CHEMBL3885882'
            #     standard_type = 'MCHC'
            #     fully_observed_task = ChEMBLDataset(assay_chembl_id=assay_chembl_id, standard_type=standard_type)
            elif self.taskname == "branin":
                fully_observed_task = BraninTask("generated_datasets/branin/branin_unif_5000.p")
            else:
                raise NotImplementedError()

            self.max = fully_observed_task.y.max()
            self.min = fully_observed_task.y.min()

            print("=" * 20)
            print("Task name:", self.taskname, "optima:", self.optima,  "Dataset min/max: {}/{}".format(self.min, self.max))
            print("=" * 20)

    def eval(self, x):
        if self.oracle:
            if torch.is_tensor(x):
                x = x.view(1, -1)
                y = self.task.predict(x.cpu().numpy())
            else:
                y = self.task.predict(x)
        else:
            if torch.is_tensor(x):
                x = x.view(1, -1)
            else:
                x = torch.tensor(x, dtype=torch.float32)

            with torch.no_grad():
                y = self.forward_net(x.to('cuda'))
                # y = y[:,0]

            y = y * (self.max - self.min) + self.min

        if self.normalise:
            assert self.max is not None
            assert self.min is not None
            y = (y - self.min) / (self.max - self.min)
        return float(y)

    def eval_unnormalise(self, x):
        if torch.is_tensor(x):
            x = x.view(1, -1)
            y = self.task.predict(x.cpu().numpy())
        else:
            y = self.task.predict(x)

    def regret(self, x):
        return self.optima - self.eval(x)

    def reward(self, x):
        raise NotImplementedError

if __name__ == "__main__":
    import pickle as pkl 
    import os
    
    # for task in ["tfbind8", "tfbind10", "utr", "ant", "dkitty", "superconductor", "hopper"]:
    for task in ["tfbind10"]:
        func = DesignBenchFunctionWrapper(task)
        points = np.random.uniform(0, 1, size=(10, func.task.x.shape[1]))
        values = func.task.predict(points)
        print(f"Task: {task} Completed")
    
    print("All Tasks Available")


