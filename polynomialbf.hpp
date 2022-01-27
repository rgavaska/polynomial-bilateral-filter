/*
 * =====================================================================================
 *
 *       Filename:  polynomialbf.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Friday 18 August 2017 04:44:09  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef POLYNOMIALBF_HPP
#define POLYNOMIALBF_HPP

#include "opencv2/core.hpp"

using namespace cv;     /* For cv::Mat */

namespace polynomialbf
{
    void polynomialBF(const cv::Mat&,const std::string&,double,double,int,cv::Mat&);
    void validateParameters(const std::string&,const double);
    class MinMaxFilter;
    template <class mat_type> class LogicalIndexer;
    class CoefficientCalculator;
    class IntegralCalculator;
}













#endif /* POLYNOMIALBF_HPP */

