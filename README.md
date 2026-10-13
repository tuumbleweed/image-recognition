# Image Recognition
Neural network that recognizes hand-written numbers.
Created using 3Blue1Brown neural network playlist.

Best result so far - 90.17% testing accuracy.

Got images from https://yann.lecun.com/exdb/mnist/

Take files containing lists of 60k training images and labels. Train neural network using this list.
Take files containing lists of 10k testing images and labels. Measure neural network accuracy.

## Commands
```bash
cmake -B build -S . # generate makefile
cmake --build build # build using makefile
./build/ImageRecognition # run

# build
cmake -B build -S . && cmake --build build

# together
cmake -B build -S . && cmake --build build && ./build/ImageRecognition
cmake -B build -S . && cmake --build build && ./build/ImageRecognitionTest
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --gpu --cpu -l DEBUG --test 0 -n 1
cmake -B build -S . && cmake --build build && ./build/CudaInspection
# batch size of 1
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --cpu -l NOTICE --seed 50 --batch-size 1 --test 1000 --train 10000
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --gpu -l NOTICE --seed 50 --batch-size 1 --test 1000 --train 10000
# batch size of 2
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --cpu -l NOTICE --seed 50 --batch-size 2 --test 1000 --train 10000
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --gpu -l NOTICE --seed 50 --batch-size 2 --test 1000 --train 10000
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --cpu -l NOTICE --seed 50 --batch-size 2 --test 0 --train 2
cmake -B build -S . && cmake --build build && ./build/ImageRecognition --gpu -l NOTICE --seed 50 --batch-size 2 --test 0 --train 2
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
    - ~~double* next_delta = delta + max_output_size - network[i + 1].output.size(); points to wrong location.
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
- ~~Feed data in batches of N (CPU)~~
- ~~Feed data in batches of N (GPU)~~
- ~~Accumulate weight and bias changes for both cpu and gpu and print them in INFO. (for each batch of images)~~
- ~~See why they differ.~~
    - ~~After accumulating changes and averaging gradients it was mostly fixed.~~
    - ~~The small difference (around 0.20% in accuracy) between the CPU and GPU is expected and largely unavoidable due to the way floating-point arithmetic works on different platforms. The discrepancy in activations you are seeing is within a reasonable range (differences in the 6th decimal place), and should not significantly affect the overall performance of the model.~~
- ~~NOW I FIXED IT! CPU==GPU FOR SURE!~~
    - ~~So the problem was that for cpu I did use average activations network[j].output = activations_average[j]
        but for gpu I forgot so I always used last activation that was there because forward_propagation_gpu modifies gpu_network.
        So thats why when increasing batch size to 2 and above gpu results got more and more nonsensical.
        The bigger the batch - the less correct the fake "average activation" ie last activation.~~
- ~~Improve accuracy.~~
- Optimize training and testing speed.

(GPU) Got success rate 90.17% for learing rate 0.14 batch size 3 train duration: 51.12 test duration: 28.09

(CPU) Got success rate 90.17% for learing rate 0.14 batch size 3 train duration: 625.72 test duration: 42.56
