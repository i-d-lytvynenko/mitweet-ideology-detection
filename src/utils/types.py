from typing import Any

import numpy as np


arr_1d_f = np.ndarray[Any, np.dtype[np.floating[Any]]]  # pyright: ignore[reportExplicitAny]
arr_2d_f = np.ndarray[tuple[Any, Any], np.dtype[np.floating[Any]]]  # pyright: ignore[reportExplicitAny]
