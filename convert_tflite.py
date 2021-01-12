#!/usr/bin/python3
# @Time    : 11/01/2021 08:51
# @Author  : Giang(Koi)
# @Email   : terminate2593@gmail.com
# @File    : convert_tflite.py
# @Software: PyCharm
from tensorflow import lite
import tensorflow as tf
from models import CTPN


def main():
    # model = CTPN()

    try:
        # model.load_weights('model/ctpn.h5')
        model = tf.keras.models.load_model('model/ctpn.h5', custom_objects={'tf': tf})
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()
        open("tflite_test.tflite", "wb").write(tflite_model)

    except Exception as ex:
        print(ex.__str__())


if __name__ == '__main__':
    main()
