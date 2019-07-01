from .generator import CocoGenerator

def from_config(config, **kwargs):
	return CocoGenerator(config['data_dir'], config['set_name'], **kwargs)
