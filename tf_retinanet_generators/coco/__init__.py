"""
Copyright 2017-2019 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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


def from_config(config, submodels_manager, preprocess_image, **kwargs):
	generators = {}

	# Set default configuration parameters.
	config = set_defaults(config)

	# If no data dir is set ask the user for it.
	if ('data_dir' not in config) or not config['data_dir']:
		config['data_dir'] = input('Please input the COCO dataset folder:')

	num_classes = 0

	# If needed get the train generator.
	if config['train_set_name'] is not None:
		generators['train'] = CocoGenerator(config, config['data_dir'], config['train_set_name'], preprocess_image)
		num_classes = generators['train'].num_classes()

	# Disable the transformations after getting the train generator.
	config['transform_generator']     = None
	config['visual_effect_generator'] = None

	# If needed get the validation generator.
	if config['validation_set_name'] is not None:
		generators['validation'] = CocoGenerator(config, config['data_dir'], config['validation_set_name'], preprocess_image)
		num_classes = generators['validation'].num_classes()

	# If needed get the test generator.
	if config['test_set_name'] is not None:
		generators['test'] = CocoGenerator(config, config['data_dir'], config['test_set_name'], preprocess_image)
		num_classes = generators['test'].num_classes()

	generators['custom_evaluation']          = evaluate_coco
	generators['custom_evaluation_callback'] = CocoEval


	# Set up the submodels for this generator.
	assert not submodels_manager.num_classes(), "Classification already has a setup number of classes."
	assert num_classes != 0, "Got 0 classes from the generator."

	submodels_manager.create(num_classes=num_classes)

	return generators, submodels_manager.get_submodels()
