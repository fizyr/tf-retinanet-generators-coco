from .generator import CocoGenerator
from .eval      import evaluate_coco
from .eval      import CocoEval


def set_defaults(config):
	# Set defaults for train generators.
	if 'train_set_name' not in config:
		config['train_set_name'] = 'train2017'
	if config['train_set_name'] == 'none':
		config['train_set_name'] = None

	# Set defaults for validation generators.
	if 'validation_set_name' not in config:
		config['validation_set_name'] = 'val2017'
	if config['validation_set_name'] == 'none':
		config['validation_set_name'] = None

	# Set defaults for test generators.
	if 'test_set_name' not in config:
		config['test_set_name'] = 'val2017'
	if config['test_set_name'] == 'none':
		config['test_set_name'] = None

	return config


def from_config(config, preprocess_image, **kwargs):
	generators = {}

	config = set_defaults(config)

	# If no data dir is set ask the user for it.
	if ('data_dir' not in config) or not config['data_dir']:
		config['data_dir'] = input('Please input the COCO dataset folder:')

	# If needed get the train generator.
	if config['train_set_name'] is not None:
		generators['train'] = CocoGenerator(config, config['data_dir'], config['train_set_name'], preprocess_image)

	# Disable the transformations after getting the train generator.
	config['transform_generator']     = None
	config['visual_effect_generator'] = None

	# If needed get the validation generator.
	if config['validation_set_name'] is not None:
		generators['validation'] = CocoGenerator(config, config['data_dir'], config['validation_set_name'], preprocess_image)

	# If needed get the test generator.
	if config['test_set_name'] is not None:
		generators['test'] = CocoGenerator(config, config['data_dir'], config['test_set_name'], preprocess_image)

	generators['custom_evaluation']          = evaluate_coco
	generators['custom_evaluation_callback'] = CocoEval
	return generators
