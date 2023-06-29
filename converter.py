from tensorflow.keras.models import load_model
import onnx
import tf2onnx.convert
import tensorflow as tf

def convert():
    # Load the keras model
    model = load_model('C:\\Users\\flash\\OneDrive\\桌面\\LSTM_predict_therapist_trajectory\\saved_models\\LSTM_trajectory_z.h5')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 12], dtype=tf.float32)])
    def model_func(input):
        return model(input)

    concrete_func = model_func

    # Let's assume the model takes images of size 28x28x1 and the batch size is undefined
    input_signature = [tf.TensorSpec(shape=[None, None, 12], dtype=tf.float32)]

    # onnx_model, _ = tf2onnx.convert.from_function(model, input_signature=input_signature)

    onnx_model, _ = tf2onnx.convert.from_function(
        function=concrete_func,
        input_signature=input_signature,
        opset=13  # specify the opset version you want to use
    )
    onnx.save(onnx_model, 'C:\\Users\\flash\\OneDrive\\桌面\\LSTM_predict_therapist_trajectory\\converted_models\\LSTM_trajectory_z'
                          '.onnx')



if __name__ == '__main__':
    convert()