import numpy as np

from utils.norm import Linf
from utils.types import Loss, Data, Model, Norm


def fast_gradient_sign_method(loss: Loss, model: Model, data: Data, xi: float) -> Data:
    return fast_gradient_dual_norm_method(loss, model, Linf, data, xi)


def fast_gradient_dual_norm_method(loss: Loss, model: Model, norm: Norm, data: Data, xi: float) -> Data:
    x = model.get_x(data.get("x").copy())
    x = x + xi * norm.dual_dx(loss.dx(model, data))
    return {"x": x, "y": data.get("y")}


def projected_gradient_ascent(loss: Loss, model: Model, proj: Norm, data: Data, xi: float, alpha: float,
                              k: int) -> Data:
    x0 = model.get_x(data.get("x").copy())
    x = x0.copy()
    for i in range(k):
        _data = {'x': x, 'y': data.get('y')}
        y = x + alpha * loss.dx(model, _data)
        x = proj.proj(y, x0, xi)
    return {"x": x, "y": data.get("y")}
