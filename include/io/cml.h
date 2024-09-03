#ifndef CML_H
#define CML_H

#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::variables_map parse_command_line(int argc, char *argv[]);

#endif // CML_H