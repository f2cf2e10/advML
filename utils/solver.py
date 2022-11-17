from typing import List, Tuple
import numpy as np
from utils.types import Model, Loss, Data, Norm


def gd(loss: Loss, model0: Model, data: List[Data], tol: float, eta: float = 0.05, proj: Norm = None,
       constraint: float = 1.0) -> Tuple[Model, List[float]]:
    training_loss = []
    delta = np.inf
    model_ = model0.copy()
    model = model0.copy()
    i = 0
    while delta > tol:
        theta = proj.proj(model_.get_theta() - eta * loss.dtheta_batch(model, data), constraint) \
            if proj != None else model_.get_theta() - eta * loss.dtheta_batch(model, data)
        model.set_theta(theta)
        delta = np.abs(loss.f_batch(model, data) - loss.f_batch(model_, data))
        training_loss += [loss.f_batch(model, data)]
        model_ = model.copy()
        print(i, delta)
        i += 1
    return model, training_loss

