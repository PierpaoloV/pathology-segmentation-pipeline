# torch_processor_pier is kept as an alias for backwards compatibility.
# torch_processor now handles both single-model and ensemble inference automatically.
from .torch_processor import torch_processor as torch_processor_pier  # noqa: F401
