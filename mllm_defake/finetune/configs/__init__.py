from pathlib import Path


this_dir = Path(__file__).parent.resolve()
configs = list(this_dir.glob("*.yml")) + list(this_dir.glob("*.yaml"))
SUPPORTED_CONFIGS = dict()

for config in configs:
    config_name = config.stem
    SUPPORTED_CONFIGS[config_name] = str(config.resolve())

__all__ = ["SUPPORTED_CONFIGS"]
