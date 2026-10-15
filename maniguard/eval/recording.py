"""Optional observer loading for passive rollout recording."""
import importlib

RECORDING_API_VERSION = 1


def make_observer(cfg,scene):
    factory=getattr(cfg,'recording_factory',None)
    output=getattr(cfg,'recording_output_dir',None)
    if not factory:
        if output:raise ValueError('recording_output_dir requires recording_factory')
        return None
    if not output or ':' not in factory:raise ValueError('Provide recording output and module:function factory')
    module,name=factory.rsplit(':',1)
    return getattr(importlib.import_module(module),name)(cfg=cfg,scene=scene)
