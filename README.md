# Entropy
code for calculating image entropy

* python code is under `python` folder

* c++ code is under `c++` folder

* test images are under `imgs` folder

## Python codes(Python 2 & Python 3)
### `entropy.py`
It contains the basic code for 1-D image entropy.

### `entropy2d.py`
It contains the basic code for 2-D image entropy.The code will run very slowly,it is kept just for beginners to learn the basic principles and metholds.Remember DO NOT use it for practical circumstance.

### `entropy2dSpeedUp.py`
It contains optimized code for better performance.And it is very recommanded to use it in practical circumstance.
optimization this version code, compared runtime in image size 911x480 as following:
#ori
H: 13.936284981535263
runtime: 8.624836444854736 s
#opti 
H: 13.936284981535266
runtime: 1.96828293800354 s

### `entropy2dCounter.py`
It is the same with `entropy2dSpeedUp.py`,I  just add some time evaluating code into it.

### `entropy2dMyFij.py`
I write the code for calculating Fij with my algorithm in this file.The performance is a little slower than `entropy2dSpeedUp.py`.But the algorithm is very useful in c++ codes.

## C++ codes
The whole project is arranged by CMake. So if you want to run it,you must have a CMake environment on your computer.The only dependce library is OpenCV.The exe file is under `cmake-build-debug` folder,name is `entropy2d`.

## Tips
After many tests,C++ project has a best performance,which is shown as below.

![](https://github.com/zhaoxuhui/Entropy/blob/master/compare.png)

So if you can satisify the requirements for c++ project mentioned above,it's best to use it.If not, you can choose the python code `entropy2dSpeedUp.py`,it will provide almost the greatest performance.

Last but not least,if you have any thing to say, don't forget to open an issue or connect me.
