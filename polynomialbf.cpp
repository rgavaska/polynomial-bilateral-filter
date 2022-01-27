/*
 * =====================================================================================
 *
 *       Filename:  polynomialbf.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  Friday 18 August 2017 09:58:29  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef POLYNOMIALBF_CPP
#define POLYNOMIALBF_CPP

#include<iostream>
#include<cmath>
#include "opencv2/core.hpp"
#include "polynomialbf.hpp"
#include "logicalindexer.cpp"
#include "minmaxfilter.cpp"
#include "integralcalculator.cpp"
#include "coefficientcalculator.cpp"

#ifdef ROUND2INT
#undef ROUND2INT
#endif
#define ROUND2INT(x) (static_cast<int>((x) + 0.5))


void polynomialbf::polynomialBF(
        const cv::Mat& inimg,        /* Input image                             */
        const std::string& kernel,   /* "gaussian" or "box"                     */
        double sigma_s,              /* Spatial filter parameter                */
        double sigma_r,              /* Standard deviation of range kernel      */
        int N,                       /* Degree of polynomial                    */
        cv::Mat& out                 /* cv::Mat object to store output image    */
        )
{
    polynomialbf::validateParameters(kernel,sigma_r);

    cv::Mat f;
    inimg.convertTo(f,CV_64FC1);
    f = f/255.0;
    sigma_r /= 255.0;

    polynomialbf::LogicalIndexer<double> li;
    polynomialbf::MinMaxFilter mmf;
    polynomialbf::IntegralCalculator ic;
    polynomialbf::CoefficientCalculator cc;

    int w;      // Kernel window size
    w = (kernel=="gaussian") ? ROUND2INT(6*sigma_s) : ROUND2INT(2*sigma_s);
    w = (w%2==1) ? w : w+1;

    // Compute local histogram ranges
    cv::Mat Alpha,Beta;
    mmf.minFilter(f,w,Alpha);
    mmf.maxFilter(f,w,Beta);
    out = Beta.clone();

    // Extract mask for pixels with nontrivial local histogram ranges
    cv::Mat mask,Alpha_mask,Beta_mask,f_mask;
    li.getMask(Alpha,"!=",Beta,mask);
    li.getMaskedValues(Alpha,mask,Alpha_mask);
    li.getMaskedValues(Beta,mask,Beta_mask);
    li.getMaskedValues(f,mask,f_mask);

    // Calculate a and b
    cv::Mat a,b;
    a = 1.0 / (Beta_mask - Alpha_mask);
    b = -Alpha_mask / (Beta_mask - Alpha_mask);

    // Calculate polynomial coefficients
    cv::Mat C;
    cc.computeCoefficients(f,Alpha,Beta,kernel,sigma_s,N,C);

    // Calculate definite integrals
    cv::Mat F;
    cv::Mat v0 = a.mul(f_mask) + b;
    cv::Mat lambda = 1.0 / (2*sigma_r*sigma_r*a.mul(a));
    ic.computeIntegrals(N+1,lambda,v0,F);

    // Calculate numerator & denominator
    cv::Mat num = cv::Mat::zeros(1,f_mask.cols,CV_64FC1);
    cv::Mat den = cv::Mat::zeros(1,f_mask.cols,CV_64FC1);
    for(int k=0; k<=N; k++)
    {
        num = num + C.row(k).mul(F.row(k+1));
        den = den + C.row(k).mul(F.row(k));
    }

    // Set pixel values in output
    out.create(f.rows,f.cols,CV_64FC1);
    li.setMaskedValues(out,mask,((num/den)-b)/a);
    li.setMaskedValues(out,mask,f,(uint8_t)0);
    out = 255*out; 
}

void polynomialbf::validateParameters(
        const std::string& kernel,
        const double sigma_r
        )
{
    if(kernel!="gaussian" && kernel!="box")
    {
        throw std::invalid_argument("Filter type must be either \"gaussian\" or \"box\"");
    }
    if(sigma_r<=1)
    {
        std::cout << "Warning: The scale of sigma_r should be [0,255]" << std::endl;
    }
}

#endif  /* POLYNOMIALBF_CPP */

