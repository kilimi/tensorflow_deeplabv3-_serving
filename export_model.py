import os
import tensorflow as tf
import shutil

from tensorflow.python.tools import freeze_graph
from deeplab import common
from deeplab import input_preprocess
from deeplab import model
from PIL import Image

slim = tf.contrib.slim

EXPORT_PATH = "frozen/1"
CHECKPOINT_PATH = "training" 
NUMBER_OF_CLASSES = 5
CROP_SIZE = [513, 513]
ATROUS_RATES = [6, 12, 18]
OUTPUT_STRIDE = 16
INFERENCE_SCALES = [1.0]
ADD_FLIPPED_IMAGES = False
IMAGE_PYRAMID = None
#####################################################################################
def preprocess_image(image_buffer):
    decode = tf.image.decode_png(image_buffer, channels=3)
    original_image, image, _ = input_preprocess.preprocess_image_and_label(
       decode,
       label=None,
       crop_height=CROP_SIZE[0],
       crop_width=CROP_SIZE[1],
       min_resize_value=None,
       max_resize_value=None,
       resize_factor=None,
       ignore_label=255,
       is_training=False,
       model_variant="xception_65")

    #image = tf.expand_dims(image, 0)

    return image
#####################################################################################
def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', EXPORT_PATH)


  with tf.Session(graph=tf.Graph()) as sess:
    # placeholder for receiving the serialized input image
    '''serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'image': tf.FixedLenFeature(shape=[], dtype=tf.float32), }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)


    # reshape the input image to its original dimension
    tf_example['image'] = tf.reshape(tf_example['image'], (1, CROP_SIZE[0], CROP_SIZE[1], 3))
    input_tensor = tf.identity(tf_example['image'], name='image')
    print("====> INPUT TENSOR", input_tensor) '''

    serialized_tf_example = tf.placeholder(tf.string, name='input_image')
    feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    pngs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, pngs, dtype=tf.float32)


    model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: NUMBER_OF_CLASSES},
            crop_size=CROP_SIZE,
            atrous_rates=ATROUS_RATES,
            output_stride=OUTPUT_STRIDE)._replace(
            model_variant="xception_65",
            decoder_output_stride=4)

    predictions = model.predict_labels(
          images,
          model_options=model_options,
          image_pyramid= IMAGE_PYRAMID)

    semantic_predictions = predictions[common.OUTPUT_TYPE]
    print("common.OUTPUT_TYPE: ", common.OUTPUT_TYPE)
    print("semantic_predictions: ", semantic_predictions)

    #restore model from checkpoints
    saver = tf.train.Saver()
    print("====>", CHECKPOINT_PATH)
    module_file=tf.train.latest_checkpoint(CHECKPOINT_PATH)
    saver.restore(sess, module_file)


    #remove folder if exists
    if os.path.exists(EXPORT_PATH) and os.path.isdir(EXPORT_PATH):
    	shutil.rmtree(EXPORT_PATH)
    builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_PATH)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(pngs)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(semantic_predictions)
    print("---->tensor info build")

    signature_def_map={
        "predict_image":tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"image": tensor_info_x},
            outputs={"seg":  tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    }
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map=signature_def_map)
    builder.save()

if __name__ == '__main__':
  tf.app.run()
