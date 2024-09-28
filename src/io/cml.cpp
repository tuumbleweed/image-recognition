#include "io/cml.h"

#include <iostream>

po::variables_map parse_command_line(int argc, char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("log,l", po::value<std::string>()->default_value("NOTICE"), "Choose log level")
        ("config,c", po::value<std::string>()->default_value("cfg/config.ini"), "Config file")
        ("train", po::value<size_t>()->default_value(-1),"How many images to train on, default is the whole data set")
        ("test", po::value<size_t>()->default_value(-1), "How many images to test on, default is the whole data set")
        ("batch-size,b", po::value<size_t>()->default_value(1), "How many images to train on at a time, default is 1")
        ("cpu", po::bool_switch()->default_value(false), "Use cpu")
        ("gpu", po::bool_switch()->default_value(false),"Use gpu")
        ("learning-rate,lr", po::value<double>()->default_value(0.05f), "Learning rate")
        ("seed,s", po::value<unsigned int>()->default_value(0), "Seed for initializing network weights and biases. Set to 0 to use random seed.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
    }

    return vm;
}
