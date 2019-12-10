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

from pycocotools.cocoeval import COCOeval

import tensorflow as tf
import numpy as np
import json

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

def get_coco_evaluation(use_mask=False):
	def _evaluate_coco(generator, model, threshold=0.05):
		""" Use the pycocotools to evaluate a COCO model on a dataset.
		Args
			generator : The generator for generating the evaluation data.
			model     : The model to evaluate.
			threshold : The score threshold to use.
		"""
		# Start collecting results.
		results = []
		image_ids = []
		for index in progressbar.progressbar(range(generator.size()), prefix='COCO evaluation: '):
			image = generator.load_image(index)
			image_shape = image.shape
			image = generator.preprocess_image(image)
			image, scale = generator.resize_image(image)

			if tf.keras.backend.image_data_format() == 'channels_first':
				image = image.transpose((2, 0, 1))

			# If mask is used the outputs are in different positions.
			offset = 0
			if use_mask:
				import cv2
				from pycocotools import mask as mask_utils
				offset = 1

			# Run network.
			outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
			boxes   = outputs[-3 - offset]
			scores  = outputs[-2 - offset]
			labels  = outputs[-1 - offset]

			# Correct boxes for image scale.
			boxes /= scale

			# Change to (x, y, w, h) (MS COCO standard).
			boxes[:, :, 2] -= boxes[:, :, 0]
			boxes[:, :, 3] -= boxes[:, :, 1]

			elements = [boxes[0], scores[0], labels[0]]

			if use_mask:
				masks = outputs[-1]
				elements.append(masks[0])

			# Compute predicted labels and scores.
			for output in zip(*(tuple(elements))):
				box   = output[0]
				score = output[1]
				label = output[2]

				# Scores are sorted, so we can break.
				if score < threshold:
					break

				# Append detection for each positively labeled class.
				image_result = {
					'image_id'    : generator.image_ids[index],
					'category_id' : generator.label_to_coco_label(label),
					'score'       : float(score),
					'bbox'        : box.tolist(),
				}

				if use_mask:
					# Get the mask.
					mask = output[3]
					mask = mask.astype(np.float32)

					# Box (x, y, w, h) as one int vector.
					b = box.astype(int)

					# Resize and binarize mask.
					mask = cv2.resize(mask[:, :, label], (b[2], b[3]))
					mask = (mask > 0.5).astype(np.uint8)

					# Encode as RLE.
					segmentation = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
					segmentation[b[1]:b[1] + b[3], b[0]:b[0] + b[2]] = mask
					segmentation = mask_utils.encode(np.asfortranarray(segmentation))

					image_result['segmentation'] = segmentation

					# Convert byte to str to write in json (in Python 3).
					if not isinstance(image_result['segmentation']['counts'], str):
						image_result['segmentation']['counts'] = image_result['segmentation']['counts'].decode()

				# Append detection to results.
				results.append(image_result)

			# Append image to list of processed images.
			image_ids.append(generator.image_ids[index])

		if not len(results):
			return

		# Write output.
		json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
		json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

		# Load results in COCO evaluation tool.
		coco_true = generator.coco
		coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))

		# Run COCO evaluation.
		coco_eval = COCOeval(coco_true, coco_pred, 'segm' if use_mask else 'bbox')
		coco_eval.params.imgIds = image_ids
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()
		return coco_eval.stats

	return _evaluate_coco


def get_coco_evaluation_callback(use_mask=False):
	class CocoEval(tf.keras.callbacks.Callback):
		""" Performs COCO evaluation on each epoch.
		"""
		def __init__(self, generator, tensorboard=None, threshold=0.05):
			""" CocoEval callback intializer.
			Args
				generator   : The generator used for creating validation data.
				tensorboard : If given, the results will be written to tensorboard.
				threshold   : The score threshold to use.
			"""
			self.generator   = generator
			self.threshold   = threshold
			self.tensorboard = tensorboard
			self.use_mask    = use_mask

			super(CocoEval, self).__init__()

		def on_epoch_end(self, epoch, logs=None):
			logs = logs or {}

			coco_tag = [
				'AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
				'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
				'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
				'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
				'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
				'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
				'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
				'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
				'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
				'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
				'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
				'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
			]
			evaluate_coco = get_coco_evaluation(use_mask=use_mask)
			coco_eval_stats = evaluate_coco(self.generator, self.model,threshold=self.threshold)
			if coco_eval_stats is not None and self.tensorboard is not None and self.tensorboard.writer is not None:
				summary = tf.Summary()
				for index, result in enumerate(coco_eval_stats):
					summary_value = summary.value.add()
					summary_value.simple_value = result
					summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
					self.tensorboard.writer.add_summary(summary, epoch)
					logs[coco_tag[index]] = result

	return CocoEval
