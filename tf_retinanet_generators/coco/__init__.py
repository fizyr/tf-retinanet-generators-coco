from .generator import CocoGenerator
from .eval      import evaluate_coco
from .eval      import CocoEval


def from_config(config, **kwargs):
	generators = {}
	if 'train_set_name' in config and config['train_set_name'] is not None:
		generators['train'] = CocoGenerator(config['data_dir'], config['train_set_name'], **kwargs)

	if 'validation_set_name' in config and config['validation_set_name'] is not None:
		generators['validation'] = CocoGenerator(config['data_dir'], config['validation_set_name'], **kwargs)

	if 'test_set_name' in config and config['test_set_name'] is not None:
		generators['test'] = CocoGenerator(config['data_dir'], config['test_set_name'], **kwargs)

	generators['custom_evaluation']          = evaluate_coco
	generators['custom_evaluation_callback'] = CocoEval
	return generators
