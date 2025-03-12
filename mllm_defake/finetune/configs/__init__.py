import os


this_dir = os.path.dirname(__file__)
configs = os.listdir(this_dir)
SUPPORTED_CONFIGS = dict()

for config in configs:
    if config.endswith(".yml") or config.endswith(".yaml"):
        config_name = config.split(".")[0]
        SUPPORTED_CONFIGS[config_name] = os.path.join(this_dir, config)


settings_dir = os.path.join(this_dir, "deepspeed")
settings = os.listdir(settings_dir)
DEEPSPEED_SETTINGS = dict()

for setting in settings:
    if not setting.endswith(".json"):
        continue
    setting_name = setting.split(".")[0]
    DEEPSPEED_SETTINGS[setting_name] = os.path.join(this_dir, "deepspeed", setting)


__all__ = ["SUPPORTED_CONFIGS", "DEEPSPEED_SETTINGS"]
