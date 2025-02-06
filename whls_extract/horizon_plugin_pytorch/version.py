__version__ = '2.5.9+cu118.torch230'
git_version = 'Unknown'
torch_version = '2.3.0'
torchvision_version = '0.18.0'
from .utils.version_helper import check_version
check_version('torch', torch_version, True, True)
