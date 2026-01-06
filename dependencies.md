# Dependencies

The code requires many python modules, and Google's Tesseract engine.

## Python Modules

Here is a list of everything imported in the API code:

```python
# utils.py
from difflib import SequenceMatcher
import string
from typing import Optional
from matplotlib.figure import Figure
from numpy import ndarray
from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# pipeline.py
from dataclasses import dataclass, field, asdict
import json
import pytesseract
from utils import *
from typing import Any
```

## Tesseract

The Tesseract **engine** can be downloaded by following
[these](https://github.com/UB-Mannheim/tesseract/wiki) instructions.
I am not sure how to include the Tesseract engine inside the API. If it is not possible, I can encourage users of
the API to download and install the engine before using the API, otherwise it won't work.