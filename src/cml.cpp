#include <iostream>

#include "cml.h"

po::variables_map parse_command_line(int argc, char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("number,n", po::value<size_t>()->default_value(-1), "How many images to train on, default is the whole data set")
        ("config,c", po::value<std::string>()->default_value("cfg/config.ini"), "Config file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
    }

    return vm;
}
