import datetime
import warnings

from mmtrack.utils import register_all_modules as register_all_mmtrack_modules
from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmtrack into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmtrack default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmtrack`, and all registries will build modules from mmtrack's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import shift_tta.fileio  # noqa: F401,F403
    import shift_tta.datasets  # noqa: F401,F403
    import shift_tta.models  # noqa: F401,F403

    # register parent modules
    register_all_mmtrack_modules(init_default_scope=False)

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('shift_tta')
        if never_created:
            DefaultScope.get_instance('shift_tta', scope_name='shift_tta')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'shift_tta':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "shift_tta", '
                          '`register_all_modules` will force the current'
                          'default scope to be "shift_tta". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'shift_tta-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='shift_tta')