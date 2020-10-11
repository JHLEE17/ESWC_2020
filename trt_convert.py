import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


input_saved_model_dir = '/home/ubuntu/jongho/capstone/Multi-Camera-Live-Object-Tracking-master/traffic_counting_jh/model_data/saved_model/w3000-pb'
output_saved_model_dir = '/home/ubuntu/jongho/capstone/Multi-Camera-Live-Object-Tracking-master/traffic_counting_jh/model_data/trt_output/'

converter = trt.TrtGraphConverter(input_saved_model_dir=input_saved_model_dir, max_workspace_size_bytes=(11<32), precision_mode="FP16", maximum_cached_engines=100)

converter.convert()
converter.save(output_saved_model_dir)





