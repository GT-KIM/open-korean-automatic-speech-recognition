import torch

from abc import ABC, abstractmethod


class BaseASRInferenceModel(ABC):
    def __init__(self):
        self.TORCH_DTYPE = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def initialize_processor(self):
        pass

    def inference_sample(self):
        pass
