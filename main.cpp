/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Friday 18 August 2017 03:52:17  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#include<iostream>
#include<chrono>
#include<string>
#include<stdexcept>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "polynomialbf.cpp"
#include "exactbf.cpp"

namespace chr = std::chrono;

struct Parameters
{
    std::string input_file;
    double sigma_s;
    double sigma_r;
    std::string filtertype = "box";
    int N;
    std::string output_file_exact;
    std::string output_file_poly;
};

void parseArguments(char *argv[], int argc, struct Parameters& params);

int main(int argc, char* argv[])
{
    // Parse arguments
    struct Parameters params;
    try
    {
        parseArguments(argv, argc, params);
    }
    catch(const std::invalid_argument& e)
    {
        std::cerr << e.what();
        std::cout << "Syntax:" << std::endl;
        std::cout << argv[0] << " <inputfile> [sigma_s] [sigma_r] [N]" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat f,Bf_pol,Bf_exact;
    f = cv::imread(params.input_file,IMREAD_GRAYSCALE);

    // Polynomial BF
    std::cout << std::endl << "Running polynomial bilateral filter ..." << std::endl;
    chr::steady_clock::time_point begin1 = chr::steady_clock::now();
    polynomialbf::polynomialBF(f,
                               params.filtertype,
                               params.sigma_s,
                               params.sigma_r,
                               params.N,
                               Bf_pol);
    chr::steady_clock::time_point end1 = chr::steady_clock::now();
    std::cout << "Done." << std::endl;
    std::cout << "Duration of polynomial BF = " << chr::duration_cast<chr::milliseconds>(end1 - begin1).count() << " millisec" << std::endl;

    // Exact BF
    std::cout << std::endl << "Running exact bilateral filter ..." << std::endl;
    chr::steady_clock::time_point begin2 = chr::steady_clock::now();
    exactbf::exactBF(f,
                     params.filtertype,
                     params.sigma_s,
                     params.sigma_r,
                     "symmetric",
                     Bf_exact);
    chr::steady_clock::time_point end2 = chr::steady_clock::now();
    std::cout << "Done." << std::endl;
    std::cout << "Duration of exact BF = " << chr::duration_cast<chr::milliseconds>(end2 - begin2).count() << " millisec" << std::endl;

    // Calculate PSNR
    double psnr = exactbf::psnr_db(Bf_exact,Bf_pol);
    std::cout << std::endl << "PSNR = " << psnr << " dB" << std::endl << std::endl;

    // Write filtered images
    cv::imwrite(params.output_file_poly,Bf_pol);
    cv::imwrite(params.output_file_exact,Bf_exact);
    return EXIT_SUCCESS;
}


void parseArguments(char *argv[], int argc, struct Parameters& params)
{
    if(argc==1)
        throw std::invalid_argument("Input file not specified\n");
    else
    {
        // Define default values
        double default_sigma_s = 3;
        double default_sigma_r = 30.0;
        int default_N = 2;

        // Read and set user-supplied values
        params.input_file = std::string(argv[1]);
        
        if(argc >= 3)
            params.sigma_s = std::stod(std::string(argv[2]));
        else
            params.sigma_s = default_sigma_s;

        if(argc >= 4)
            params.sigma_r = std::stod(std::string(argv[3]));
        else
            params.sigma_r = default_sigma_r;

        if(argc >= 5)
            params.N = std::stoi(std::string(argv[4]));
        else
            params.N = default_N;

        std::string rawname = params.input_file.substr(0, params.input_file.find_last_of("."));
        params.output_file_exact = rawname + "_exact.png";
        params.output_file_poly = rawname + + "_poly.png";
    }

    // Print values
    std::cout << "Parameters:" << std::endl;
    std::cout << "Input image: " << params.input_file << std::endl;
    std::cout << "Spatial width (sigma_s): " << params.sigma_s << std::endl;
    std::cout << "Range width (sigma_r): " << params.sigma_r << std::endl;
    std::cout << "Polynomial degree: " << params.N << std::endl;
}

