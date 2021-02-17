#!/usr/bin/env bash

# This script installs deepstream on Jetson deveices

echo " " &&  echo "Installing DeepStream ..." && echo " "
sleep 2

echo " " &&  echo "Installing DeepStream dependecies..." && echo " "
sleep 1

cd 
apt install \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1

echo " " &&  echo "Install NVIDIA V4L2 GStreamer plugin..." && echo " "

echo "deb https://repo.download.nvidia.com/jetson/common r$JETPACK_VERSION main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
echo "deb https://repo.download.nvidia.com/jetson/$PLATFORM r$JETPACK_VERSION main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

apt update
sleep 1

apt install --reinstall nvidia-l4t-gstreamer

echo " " &&  echo "Install the DeepStream SDK..." && echo " "
sleep 2

tar -xvf $DEEPSTREAM_SDK_TAR_PATH -C /
cd /opt/nvidia/deepstream/deepstream-5.0
./install.sh
ldconfig

echo " " &&  echo "Boost the clocks..." && echo " "
sleep 2

nvpmodel -m 0
jetson_clocks

echo " " &&  echo "Install Python Bindings..." && echo " "
sleep 2

cd ~
apt-get install python3-dev libpython3-dev
apt-get install python-gi-dev
export GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"
export GST_CFLAGS="-pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include"
git clone https://github.com/GStreamer/gst-python.git
cd gst-python
git checkout 1a8f48a
./autogen.sh PYTHON=python3
./configure PYTHON=python3
make
sudo make install

echo " " &&  echo "Cloning Python sample applications..." && echo " "
sleep 2

cd
cd /opt/nvidia/deepstream/deepstream-5.0/sources/
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

echo " " &&  echo "DeepStream Inatllation Completed..." && echo " "
