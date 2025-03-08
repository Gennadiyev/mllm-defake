import os


this_dir = os.path.dirname(__file__)
configs = os.listdir(this_dir)
SUPPORTED_CONFIGS = dict()

for config in configs:
    if not config.endswith('.yml'):
        continue
    config_name = config.split('.')[0]
    SUPPORTED_CONFIGS[config_name] = os.path.join(this_dir, config)