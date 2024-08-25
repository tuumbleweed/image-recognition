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
- ~~Cost function~~
- ~~Backpropagation~~
- ~~Update weights and biases~~
- ~~Training loop~~
- ~~Testing using test images~~
- ~~It isn't training correctly, stuck at 0.9 cost~~
- ~~Experiment with learning rate (Success rate of 60-80% is achieved with learning rate from 0.05 to 0.08)~~
- ~~GPU training~~
- GPU testing
- Stop moving data from gpu to host with each image (only after N images)
- Feed data in batches of N
