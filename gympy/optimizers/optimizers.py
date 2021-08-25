import autograd.numpy as np
from pydantic import BaseModel

class GradientDescent(BaseModel):
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0
    def update(self, weigths, grads):
        params_list =[]
        for w, g in zip(weigths, grads):
            params = w - self.learning_rate * g
            params_list.append(params)
        return params_list
