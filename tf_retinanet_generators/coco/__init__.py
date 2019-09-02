from .generator import CocoGenerator
from .eval      import evaluate_coco
from .eval      import CocoEval


def from_config(config, preprocess_image, **kwargs):
	generators = {}

	# If needed get the train generator.
	if 'train_set_name' in config and config['train_set_name'] is not None:
		generators['train'] = CocoGenerator(config, config['data_dir'], config['train_set_name'], preprocess_image)

	# Disable the transformations after getting the train generator.
	config['transform_generator']     = None
	config['visual_effect_generator'] = None

	# If needed get the validation generator.
	if 'validation_set_name' in config and config['validation_set_name'] is not None:
		generators['validation'] = CocoGenerator(config, config['data_dir'], config['validation_set_name'], preprocess_image)

	# If needed get the test generator.
	if 'test_set_name' in config and config['test_set_name'] is not None:
		generators['test'] = CocoGenerator(config, config['data_dir'], config['test_set_name'], preprocess_image)

	generators['custom_evaluation']          = evaluate_coco
	generators['custom_evaluation_callback'] = CocoEval
	return generators
