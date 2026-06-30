import numpy as np


def hsva_array_to_rgba(hsva: np.ndarray) -> np.ndarray:
    in_h = hsva[:, 0]
    in_s = hsva[:, 1]
    in_v = hsva[:, 2]
    in_a = hsva[:, 3]

    out = np.zeros_like(hsva, dtype=np.uint8)
    out[:, 3] = in_a * 255

    h = np.where(in_h == 1.0, 0.0, in_h)
    f = h * 6
    i = np.floor(h * 6)
    f -= i

    i = i.astype(np.uint8)

    w = (255 * (in_v * (1.0 - in_s))).astype(np.uint8)
    q = (255 * (in_v * (1.0 - in_s * f))).astype(np.uint8)
    t = (255 * (in_v * (1.0 - in_s * (1.0 - f)))).astype(np.uint8)
    v = (255 * in_v).astype(np.uint8)

    out[:, 0] = np.choose(i, [v, q, w, w, t, v])
    out[:, 1] = np.choose(i, [t, v, v, q, w, w])
    out[:, 2] = np.choose(i, [w, w, t, v, v, q])

    return out
