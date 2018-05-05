used dataset:
	For the simplification of the image loading I used the same images in .pgm format P2 subformat (instead of the original P5). This basically contains the whole image as header information + list of pixel values in ascii encoding, instead the original which does the same in binary. After loading the images the difference between the source format is indistinguishable. I included both the new (P2 folder) and old (P5 folder) datasets. The data folder contains the P2 ones as well. Using P2 .pgm files does not require any modifications to the original OpenCV code.
	In the input folder I have also put two training/test files, these were the basis for the measurements. Be advised that the order of the images in them are shuffled. The P2 images also have different order / naming as the P5 images. You can get the label for an image with the following calculation: (x-1)//10+1 where x is the id in the name of the image, and // is the integer (truncating) division.
	
measurements:
	I used to setups:
		280 training images, 120 images to classify, use enough principal components to have 99% confidence, avg of 100 runs on the same random data
		390 training images,  10 images to classify, use enough principal components to have 99% confidence, avg of 100 runs on the same random data
	
	The results were the following:
		280/120 GPU:
		
			avg load time: 5.6s
			avg training time: 2.9s
			avg classification time: 0.36s
			avg total time: 8.9s
			(accuracy: 0.917)
		
		280/120 CPU:
		
			avg load time: 0.3s
			avg training time: 2.7s
			avg classification time: 0.0s
			avg total time: 3.0s
			(accuracy: 0.917)
		
		390/10 GPU:
		
			avg load time: 6.5s
			avg training time: 4.4s
			avg classification time: 0.3s
			avg total time: 11.2s
			(accuracy: 1.0)
		
		390/10 CPU:
		
			avg load time: 0.3s
			avg training time: 5.2s
			avg classification time: 0.0s
			avg total time: 5.5s
			(accuracy: 1.0)
			
	The load time only means the time we need to read the images from the files to the memory of the computer (not the GPU).
	Since clearly this part was making the biggest difference, I tried using the OpenCV provided functions too for image loading. I only tested this on the second setup (because in the first setup it is clear that the CPU version takes the prime). The results were the following:
	
		390/10 OpenCV/GPU:
		
			avg load time: 0.3s
			avg training time: 4.2s
			avg classification time: 0.3s
			avg total time: 4.8s
			(accuracy: 1.0)
	
compilation:
	I used a Windows machine for development, on which I used the following commands to compile the original version:
	
		"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.11
		nvcc -c "C:\Users\username\cuda-opencv-eigenfaces\calculations.cu" -o "C:\Users\username\Desktop\cuda-opencv-eigenfaces\calculations.obj" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" --cl-version=2017
		cl /EHsc /W4 /Fepca.exe io.cpp main.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" cuda.lib cudart_static.lib kernel32.lib cusolver.lib calculations.obj
	
	and for the OpenCV version:
	
		"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.11
		nvcc -c "C:\Users\username\cuda-opencv-eigenfaces\calculations.cu" -o "C:\Users\username\cuda-opencv-eigenfaces\calculations.obj" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" --cl-version=2017
		cl /EHsc /W4 /Fepca.exe io.cpp main.cpp /I "C:\Libs\OpenCV3.3.0\include" /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" /LIBPATH:"C:\Libs\OpenCV3.3.0\x64\vc15\lib\" opencv_world330.lib cuda.lib cudart_static.lib kernel32.lib cusolver.lib calculations.obj
	
	For CUDA 9.0 these require 14.11 toolset version for Visual Studio 2017.
	
	The above commands should flawlessly translate to Linux like this:
	for original
	
		nvcc -c ./calculations.cu -o ./calculations.o -L/usr/local/cuda/include
		g++ io.cpp main.cpp -o pca -L/usr/local/cuda/lib64 -lcuda -lcudart -lcusolver calculations.o
		
	
	and for OpenCV version
	
		nvcc -c ./calculations.cu -o ./calculations.obj -L/usr/local/cuda/include
		g++ io.cpp main.cpp -o pca  -Iusr/local/include/opencv2 -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgcodecs -lcuda -lcudart -lcusolver calculations.o
	
File structure:
	It is the same scheme as in the pure OpenCV one, only all the gpu related parts are now in the calculations.cu .
	In addition to the datasets I have added the 'io' folder which contains both versions (with and without) of the image loading. Just write over the original one, rename it and use the appropriate compilation command. The default one is the OpenCV version.
	
remarks:
	As you can see around 400 training images the CUDA code starts to get ahead of the CPU version. With GPUs one has to worry about the memory limits but 400 images were not even close to my 2GB limit. (In fact I think it used < 50 MB).
	I'd like to mention again that if speed is a concern,then you could consider pretraining a model, saving it on disk (this is absolutely doable) and using that for classification. As the measurments show the classification part is really fast, most of the computation hangs on the model training.
	As I have mentioned I made a few modifications to the original project, I resubmit that too. I updated the data_struct.h (since I accidentally left some  redeclarations/unnecessary includes in there) and the calculations.cpp ( in which I reduced the size of a matrix ). The runtime measurements are based on this new version.