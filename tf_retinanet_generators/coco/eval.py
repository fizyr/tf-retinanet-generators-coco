from pycocotools.cocoeval import COCOeval

import tensorflow as tf
import numpy as np
import json

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def evaluate_coco(generator, model, threshold=0.05):
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
		image = generator.preprocess_image(image)
		image, scale = generator.resize_image(image)

		if tf.keras.backend.image_data_format() == 'channels_first':
			image = image.transpose((2, 0, 1))

		# Run network.
		boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

		# Correct boxes for image scale.
		boxes /= scale

		# Change to (x, y, w, h) (MS COCO standard).
		boxes[:, :, 2] -= boxes[:, :, 0]
		boxes[:, :, 3] -= boxes[:, :, 1]

		# Compute predicted labels and scores.
		for box, score, label in zip(boxes[0], scores[0], labels[0]):
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
	coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
	coco_eval.params.imgIds = image_ids
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()
	return coco_eval.stats


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
		self.generator = generator
		self.threshold = threshold
		self.tensorboard = tensorboard

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
		coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold)
		if coco_eval_stats is not None and self.tensorboard is not None and self.tensorboard.writer is not None:
			summary = tf.Summary()
			for index, result in enumerate(coco_eval_stats):
				summary_value = summary.value.add()
				summary_value.simple_value = result
				summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
				self.tensorboard.writer.add_summary(summary, epoch)
				logs[coco_tag[index]] = result
