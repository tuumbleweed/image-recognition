# Image Recognition


## Commands
```bash
cmake -B build -S . # generate makefile
cmake --build build # build using makefile
./build/ImageRecognition # run

# together
cmake -B build -S . && cmake --build build && ./build/ImageRecognition
cmake -B build -S . && cmake --build build && ./build/ImageRecognitionTest
```


## Things to do
- ~~Read data~~
- ~~Normalize image values from 0-255 to 0-1~~
- ~~Initialize network with random values between 0-1~~
- ~~Forward propagation~~
- ~~Add biases~~
- Loss function
- Backpropagation
- Update parameters
- Training loop
- Testing using test images
