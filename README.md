# Deepstream Facenet

Face Recognition on Jetson Nano using DeepStream and Python.

## DeepStream Installation
`install-deepstream.sh` will install DeepStream and its dependencies
1. Download DeepStream using [this link](https://developer.nvidia.com/assets/Deepstream/5.0/ga/secure/deepstream_sdk_5.0.1_x86_64.tbz2)
2. get Jetpack version 
```
$ dpkg-query --show nvidia-l4t-core
nvidia-l4t-core 32.3.1-20191209225816
```
3. export needed variables
```
export JETPACK_VERSION=32.3
export PLATFORM=<platform>
export DEEPSTREAM_SDK_TAR_PATH=<path>
```
**Where <platform> identifies the platform’s processor**:
- `t186` for Jetson TX2 series
- `t194` for Jetson AGX Xavier series or Jetson Xavier NX
- `t210` for Jetson Nano or Jetson TX1
4. running installation script
```
chmod +x install-deepstream.sh
sudo -E ./install-deepstream.sh
```
5. Making sure installation is fine by running a sample app
```
cd /opt/nvidia/deepsteream/deepstream-5.0/sources/deepstream_python_apps/apps/deepstream-test1
python3 deepstream-test1.py /opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264
```
take some time to compile the model and running the application for first time.

## App
This demo is built on top of Python sample app [deepstream-test2](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test2) 
 - Download [back-to-back-detectors](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/master/back-to-back-detectors) (the mode can detect faces). It is primary inference.
 - The secondary inference facenet engine. 
  - Note: embedding dataset (npz file) should be generate by your dataset.
 - No changes regarding the tracker.

### Steps to run the demo:

- Generate the engine file for Facenet model
  - facenet_keras.h5 can be found in the models folder. The model is taken from [nyoki-mtl/keras-facenet](https://github.com/nyoki-mtl/keras-facenet)
  - Convert facenet model to TensorRT engine using [this jupyter notebook](https://github.com/riotu-lab/tf2trt_with_onnx). The steps in the jupyter notebook is taken  from [Nvidia official tutorial](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/).  

  - when converting pb file to onnx use below command instead:
  `python -m tf2onnx.convert --input facenet.pb --inputs input_1:0[1,160,160,3] --inputs-as-nchw input_1:0 --outputs Bottleneck_BatchNorm/batchnorm_1/add_1:0 --output facenet.onnx`
  **Note: make sure to use this command --inputs-as-nchw input_1:0 while converting to ONNX to avoid having this error:**
  `Error in NvDsInferContextImpl::preparePreprocess() <nvdsinfer_context_impl.cpp:874> [UID = 2]: RGB/BGR input format specified but network input channels is not 3`

- Change the model-engine-file path to the your facenet engine file in `classifier_config.txt`.
- `python3 deepstream_test_2.py <h264_elementary_stream_contains_faces`


## Resources

You can find more resources about our face recognition work and inference results at https://www.riotu-lab.org/face/
