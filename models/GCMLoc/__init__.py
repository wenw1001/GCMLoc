"""GCMLoc networks: GCMLocMapping (offline Stage 1 + 2) and GCMLocLocalization (online Stage 2)."""
from .GCMLoc_mapping import GCMLocMapping, DEPTH_DIMS
from .GCMLoc_localization import GCMLocLocalization, TIMED_MODULES
from .checkpoint import (arch_from_config, arch_from_state_dict, resolve_arch,
                         load_checkpoint, build_mapping_from_checkpoint,
                         build_localization_from_checkpoint)

__all__ = ["GCMLocMapping", "GCMLocLocalization", "DEPTH_DIMS", "TIMED_MODULES",
           "arch_from_config", "arch_from_state_dict", "resolve_arch", "load_checkpoint",
           "build_mapping_from_checkpoint", "build_localization_from_checkpoint"]
