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

from tf_retinanet.utils.image import read_image_bgr

import os
import numpy as np

from pycocotools.coco import COCO

def get_coco_generator(base_generator):

	class CocoGenerator(base_generator):
		""" Generate data from the COCO dataset.
		See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI for more information.
		"""
		def __init__(self, config, set_name, preprocess_image):
			""" Initialize a COCO data generator.
			Args
				config : Dictionary containing information about the generator.
					It should contain:
						data_dir            : Path or list of paths to the dataset directory.
						train_set_name      : Train set name.
						validation_set_name : Validation set name.
						test_set_name       : Test set name.
						mask                : Flag to enable mask loading.
				set_name        : Name of the set to parse.
				preprocess_image: Function to preprocess images.
			"""
			self.data_dir  = config['data_dir']
			self.set_name  = set_name
			self.coco      = COCO(os.path.join(self.data_dir, 'annotations', 'instances_' + set_name + '.json'))
			self.image_ids = self.coco.getImgIds()
			self.mask      = config['mask']

			self.load_classes()

			super(CocoGenerator, self).__from_config__(config, preprocess_image=preprocess_image)

		def load_classes(self):
			""" Loads the class to label mapping (and inverse) for COCO.
			"""
			# Load class names (name -> label).
			categories = self.coco.loadCats(self.coco.getCatIds())
			categories.sort(key=lambda x: x['id'])

			self.classes             = {}
			self.coco_labels         = {}
			self.coco_labels_inverse = {}
			for c in categories:
				self.coco_labels[len(self.classes)] = c['id']
				self.coco_labels_inverse[c['id']] = len(self.classes)
				self.classes[c['name']] = len(self.classes)

			# Also load the reverse (label -> name).
			self.labels = {}
			for key, value in self.classes.items():
				self.labels[value] = key

		def size(self):
			""" Size of the COCO dataset.
			"""
			return len(self.image_ids)

		def num_classes(self):
			""" Number of classes in the dataset. For COCO this is 80.
			"""
			return len(self.classes)

		def has_label(self, label):
			""" Return True if label is a known label.
			"""
			return label in self.labels

		def has_name(self, name):
			""" Returns True if name is a known class.
			"""
			return name in self.classes

		def name_to_label(self, name):
			""" Map name to label.
			"""
			return self.classes[name]

		def label_to_name(self, label):
			""" Map label to name.
			"""
			return self.labels[label]

		def coco_label_to_label(self, coco_label):
			""" Map COCO label to the label as used in the network.
			COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
			"""
			return self.coco_labels_inverse[coco_label]

		def coco_label_to_name(self, coco_label):
			""" Map COCO label to name.
			"""
			return self.label_to_name(self.coco_label_to_label(coco_label))

		def label_to_coco_label(self, label):
			""" Map label as used by the network to labels as used by COCO.
			"""
			return self.coco_labels[label]

		def image_aspect_ratio(self, image_index):
			""" Compute the aspect ratio for an image with image_index.
			"""
			image = self.coco.loadImgs(self.image_ids[image_index])[0]
			return float(image['width']) / float(image['height'])

		def load_image(self, image_index):
			""" Load an image at the image_index.
			"""
			image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
			path       = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
			return read_image_bgr(path)

		def load_annotations(self, image_index):
			""" Load annotations for an image_index.
			"""
			# Get ground truth annotations.
			annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
			annotations     = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

			# If needed get info for masks.
			if self.mask:
				import cv2

				# Get image info.
				image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
				annotations['masks'] = []

			# Some images appear to miss annotations (like image with id 257034).
			if len(annotations_ids) == 0:
				return annotations


			# Parse annotations
			coco_annotations = self.coco.loadAnns(annotations_ids)
			for idx, a in enumerate(coco_annotations):
				# Some annotations have basically no width / height, skip them.
				if a['bbox'][2] < 1 or a['bbox'][3] < 1:
					continue

				annotations['labels'] = np.concatenate([annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
				annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
					a['bbox'][0],
					a['bbox'][1],
					a['bbox'][0] + a['bbox'][2],
					a['bbox'][1] + a['bbox'][3],
				]]], axis=0)

				# If needed get annotations for masks.
				if self.mask:
					if 'segmentation' not in a:
						raise ValueError('Expected \'segmentation\' key in annotation, got: {}'.format(a))

					mask = np.zeros((image_info['height'], image_info['width'], 1), dtype=np.uint8)
					for seg in a['segmentation']:
						points = np.array(seg).reshape((len(seg) // 2, 2)).astype(int)

						# Draw mask.
						cv2.fillPoly(mask, [points.astype(int)], (1,))

					annotations['masks'].append(mask.astype(float))


			return annotations

	return CocoGenerator
