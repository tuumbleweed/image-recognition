# Image Recognition


## Commands
```bash
cmake -B build -S . # generate makefile
cmake --build build # build using makefile
./build/ImageRecognition # run

# together
cmake -B build -S . && cmake --build build && ./build/ImageRecognition
cmake -B build -S . && cmake --build build && ./build/ImageRecognitionTest
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --gpu --cpu -l DEBUG --test 0 -n 1
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
- Have an option to print all the calculations (gonna be very slow since we need to shadow all the calculations on cpu).
- See why you get only 30-40% accuracy with GPU as opposed to 60-80% with CPU
    - Layer 2 is the same
    - Layer 1 is different
    - Error is most likely in compute_delta_hidden_gpu
    - in compute_delta_hidden_gpu it multiplies wrong things when computing deltas on layers 2 and 1
        also in layer 1 lot's of values are 0 for some reason
- GPU testing
- Stop moving data from gpu to host with each image (only after N images)
- Feed data in batches of N

