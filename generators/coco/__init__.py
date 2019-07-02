from .generator import CocoGenerator

def from_config(config, **kwargs):
	generators = {}
	if config['train_set_name'] is not None:
		generators['train'] = CocoGenerator(config['data_dir'], config['train_set_name'], **kwargs)

	if config['validation_set_name'] is not None:
		generators['validation'] = CocoGenerator(config['data_dir'], config['validation_set_name'], **kwargs)
	return generators
