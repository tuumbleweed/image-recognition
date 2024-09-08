# Image Recognition
Take files containing lists of 60k training images and labels. Train neural network using this list.
Take files containing lists of 10k testing images and labels. Measure neural network accuracy.

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
- ~~It isn't training correctly, stuck at 0.9 loss~~
- ~~Experiment with learning rate (Success rate of 60-80% is achieved with learning rate from 0.05 to 0.08)~~
- ~~GPU training~~
- ~~See why you get only 30-40% accuracy with GPU as opposed to 60-80% with CPU~~
    - ~~Layer 2 is the same~~
    - ~~Layer 1 is different~~
    - ~~Error is most likely in compute_delta_hidden_gpu~~
    - ~~In compute_delta_hidden_gpu it multiplies wrong things when computing deltas on layers 2 and 1~~
    - ~~float* next_delta = delta + max_output_size - network[i + 1].output.size(); points to wrong location.
        16-10=6 next delta points to 6 instead of 0 and rest of values are 0~~
    - ~~It's fixed, all GPU calculations are equal to CPU for both layer 0 and 1~~
    - ~~Finally both CPU an GPU work the same (60-80% accuracy)
        (CPU) Got success rate 76.97% for learing rate 0.05 train duration: 59.26 test duration: 6.84
        (GPU) Got success rate 71.57% for learing rate 0.05 train duration: 142.02 test duration: 6.19~~
- ~~GPU testing~~
- ~~Set seed using --seed flag~~
- ~~GPU memory leak~~
- ~~Allocate and deallocate memory only once on GPU.~~
- ~~Reorganize project~~
- Stop moving data from gpu to host with each image (only after N images)
- Feed data in batches of N
