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

from .generator import get_coco_generator
from .eval      import get_coco_evaluation, get_coco_evaluation_callback
from tf_retinanet.utils.config import set_defaults


default_config = {
	'train_set_name'     : 'train2017',
	'validation_set_name': 'val2017',
	'test_set_name'      : 'val2017',
	'mask'               : False
}


def from_config(config, submodels_manager, preprocess_image, **kwargs):
	""" Return generators and submodels as indicated in the config.
		The number of classes (necessary for creating the classification submodel)
		is taken from the COCO generators. Hence, submodels can be initialized only after the generators.
	Args
		config : Dictionary containing information about the generators.
				 It should contain:
					data_dir            : Path to the directory where the dataset is stored.
					train_set_name      : Name of the training set.
					validation_set_name : Name of the validation set.
					test_set_name       : Name of the test set.
				 If not specified, default values indicated above will be used.
		submodel_manager : Class that handles and initializes the submodels.
		preprocess_image : Function that describes how to preprocess images in the generators.
	Return
		generators : Dictionary containing generators and evaluation procedures.
		submodels  : List of initialized submodels.
	"""
	# Set default configuration parameters.
	config = set_defaults(config, default_config)

	# If no data dir is set, ask the user for it.
	if ('data_dir' not in config) or not config['data_dir']:
		config['data_dir'] = input('Please input the COCO dataset folder:')

	generators = {}

	# We should get the number of classes from the generators.
	num_classes = 0

	# Get the generator that supports masks if needed.
	if config['mask']:
		from tf_maskrcnn_retinanet.generators import Generator
	else:
		from tf_retinanet.generators import Generator
	CocoGenerator = get_coco_generator(Generator)


	# If needed, get the train generator.
	if config['train_set_name'] is not None:
		generators['train'] = CocoGenerator(config, config['train_set_name'], preprocess_image)
		num_classes = generators['train'].num_classes()

	# Disable the transformations after getting the train generator.
	config['transform_generator_class']     = None
	config['visual_effect_generator_class'] = None

	# If needed, get the validation generator.
	if config['validation_set_name'] is not None:
		generators['validation'] = CocoGenerator(config, config['validation_set_name'], preprocess_image)
		num_classes = generators['validation'].num_classes()

	# If needed, get the test generator.
	if config['test_set_name'] is not None:
		generators['test'] = CocoGenerator(config, config['test_set_name'], preprocess_image)
		num_classes = generators['test'].num_classes()

	generators['evaluation_procedure'] = get_coco_evaluation(config['mask'])
	generators['evaluation_callback']  = get_coco_evaluation_callback(config['mask'])

	# Set up the submodels for this generator.
	assert num_classes != 0, "Got 0 classes from COCO generator."

	# Instantiate the submodels for this generator.
	submodels_manager.create(num_classes=num_classes)

	return generators, submodels_manager.get_submodels()
