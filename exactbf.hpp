

#ifndef EXACTBF_HPP
#define EXACTBF_HPP

#include "opencv2/core.hpp"

namespace exactbf
{
    void applySymmetricPadding(const cv::Mat&,int,cv::Mat&);
    void validateParameters(const std::string&,const std::string&,const double);
    void applyZeroPadding(const cv::Mat&,int,cv::Mat&);
    void gaussianKernel(int,double,cv::Mat&);
    void boxKernel(int,cv::Mat&);
    void exactBF(const cv::Mat&,const std::string&,double,double,const std::string&,cv::Mat&);
    double psnr_db(const cv::Mat&, const cv::Mat&);
}

#endif	/* EXACTBF_HPP */

