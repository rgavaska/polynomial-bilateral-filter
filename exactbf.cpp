/*
 * =====================================================================================
 *
 *       Filename:  exactBF.cpp
 *
 *    Description:  Exact bilateral filter
 *
 *        Version:  1.0
 *        Created:  Friday 18 August 2017 12:19:39  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef EXACTBF_CPP
#define EXACTBF_CPP

#include<cmath>
#include<iostream>
#include "opencv2/core.hpp"
#include "exactbf.hpp"

#ifdef ROUND2INT
#undef ROUND2INT
#endif
#define ROUND2INT(x) (static_cast<int>((x) + 0.5))


void exactbf::exactBF(
        const cv::Mat& in,              /* Input image ([0,255])                */
        const std::string& filtertype,  /* "gaussian" or "box"                  */
        double sigma_s,                 /* Spatial filter parameter             */
        double sigma_r,                 /* Standard deviation of range kernel   */
        const std::string& padtype,     /* "symmetric" or "zero"                */
        cv::Mat& out                    /* cv::Mat object to store output image */
        )
{
    exactbf::validateParameters(filtertype,padtype,sigma_r);

    cv::Mat inimg;
    in.convertTo(inimg,CV_64FC1);

    // Form spatial kernel
    cv::Mat kernel;
    int radius;
    if(filtertype=="gaussian")
    {
        radius = ROUND2INT(6*sigma_s);
        radius = (radius%2==0) ? (radius/2) : (radius-1)/2;
        exactbf::gaussianKernel(radius,sigma_s,kernel);
    }
    else if(filtertype=="box")
    {
        radius = ROUND2INT(sigma_s);
        exactbf::boxKernel(radius,kernel);
    }

    // Pad input image
    cv::Mat f;
    if(padtype=="symmetric") { exactbf::applySymmetricPadding(inimg,radius,f); }
    else if(padtype=="zero") { exactbf::applyZeroPadding(inimg,radius,f); }

    // Compute & store output values
    cv::Mat W(inimg.rows,inimg.cols,CV_64FC1);
    cv::Mat Z(inimg.rows,inimg.cols,CV_64FC1);
    cv::Mat rangekernel,nb;
    int j1,j2;
    for(j1=radius; j1<radius+inimg.rows; j1++)
    {
        for(j2=radius; j2<radius+inimg.cols; j2++)
        {
            nb = f.rowRange(j1-radius,j1+radius+1).colRange(j2-radius,j2+radius+1); // f(i-j)
            rangekernel = nb - f.at<double>(j1,j2);           // f(i-j)-f(i)
            rangekernel = rangekernel.mul(rangekernel);       // (f(i-j)-f(i))^2
            rangekernel = -0.5*rangekernel/(sigma_r*sigma_r); // (-(f(i-j)-f(i))^2)/(2*sigma_r^2)
            cv::exp(rangekernel,rangekernel);                 // exp((-(f(i-j)-f(i))^2)/(2*sigma_r^2)
            W.at<double>(j1-radius,j2-radius) = cv::sum(kernel.mul(rangekernel).mul(nb))[0];
            Z.at<double>(j1-radius,j2-radius) = cv::sum(kernel.mul(rangekernel))[0];
        }
    }
    out = W/Z;
}


void exactbf::validateParameters(
        const std::string& kernel,
        const std::string& padtype,
        const double sigma_r
        )
{
    if(kernel!="gaussian" && kernel!="box")
    {
        throw std::invalid_argument("Filter type must be either \"gaussian\" or \"box\"");
    }
    if(padtype!="symmetric" && padtype!="zero")
    {
        throw std::invalid_argument("Padding type must be either \"symmetric\" or \"zero\"");
    }
    if(sigma_r<1)
    {
        std::cout << "Warning: The scale of sigma_r should be [0,255]" << std::endl;
    }
}


void exactbf::applySymmetricPadding(const cv::Mat& in, int padsize, cv::Mat& out)
{
    out.create(in.rows+2*padsize,in.cols+2*padsize,CV_64FC1);
    cv::Mat center = out.rowRange(padsize,in.rows+padsize).colRange(padsize,in.cols+padsize);
    center = in + 0.0;
    double* rowptr;
    double* colptr;
    int r,c;

    // Apply symmetric padding to every row of input
    for(r=0; r<in.rows; r++)
    {
        rowptr = out.ptr<double>(r+padsize);
        for(c=0; c<padsize; c++)
        {
            rowptr[padsize-1-c] = rowptr[padsize+c];
            rowptr[padsize+in.cols+c] = rowptr[padsize+in.cols-1-c];
        }
    }

    // Apply symmetric padding to all columns
    for(c=0; c<out.cols; c++)
    {
        colptr = out.ptr<double>(0) + c;
        for(r=0; r<padsize; r++)
        {
            colptr[(padsize-1-r)*out.cols] = colptr[(padsize+r)*out.cols];
            colptr[(padsize+in.rows+r)*out.cols] = colptr[(padsize+in.rows-1-r)*out.cols];
        }
    }
}


void exactbf::applyZeroPadding(const cv::Mat& in, int padsize, cv::Mat& out)
{
    out = cv::Mat::zeros(in.rows+2*padsize,in.cols+2*padsize,CV_64FC1);
    cv::Mat centre = out.rowRange(padsize,padsize+in.rows).colRange(padsize,padsize+in.cols);
    in.copyTo(centre);
}


void exactbf::gaussianKernel(int radius, double sigma, cv::Mat& out)
{
    out.create(2*radius+1,2*radius+1,CV_64FC1);
    double* rowptr;
    int r,c;
    double lambda = 1/(2*sigma*sigma);
    out.at<double>(radius,radius) = 1;
    for(r=0; r<radius; r++)
    {
        rowptr = out.ptr<double>(r);
        for(c=0; c<radius; c++)
        {
            rowptr[c] = exp(-lambda*((r-radius)*(r-radius) + (c-radius)*(c-radius)));
            rowptr[2*radius-c] = rowptr[c];
        }
        rowptr[radius] = exp(-lambda*(r-radius)*(r-radius));
        out.row(r).copyTo(out.row(2*radius-r));
    }
    cv::Mat temp = out.col(radius).t();
    temp.copyTo(out.row(radius));
    double ksum = cv::sum(out)[0];
    out = out/ksum;     // Normalize
}


void exactbf::boxKernel(int radius, cv::Mat& out)
{
    out = cv::Mat::ones(2*radius+1,2*radius+1,CV_64FC1);
    out = out / (out.rows*out.cols);
}


double exactbf::psnr_db(const cv::Mat& A, const cv::Mat& B)  /* For single-channel images only */
{
    if(A.rows==B.rows && A.cols==B.cols)
    {
        cv::Mat err = A - B;                // (A-B)
        err = err.mul(err);             // (A-B)^2
        double mse = cv::sum(err)[0];   // Sum of squared errors
        mse  = mse/(A.rows*A.cols);     // Divide by no. of pixels
        double psnr = 20.0*log10(255.0) - 10.0*log10(mse);
        return(psnr);
    }
    else { throw std::invalid_argument("Sizes of input images do not match"); }
}

#endif /* EXACTBF_CPP */

