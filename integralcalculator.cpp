/*
 * =====================================================================================
 *
 *       Filename:  integralcalculator.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Thursday 10 August 2017 10:17:26  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef INTEGRALCALCULATOR_CPP
#define INTEGRALCALCULATOR_CPP

#include<cmath>
#include<cstdint>
#include<stdexcept>
#include "opencv2/core.hpp"
#include "polynomialbf.hpp"

#define N_SERIES   25
#define N_FRACTION 25
using namespace cv;

class polynomialbf::IntegralCalculator
{
    private:
        polynomialbf::LogicalIndexer<double> li;
        void binomialCoefficients(unsigned short,std::vector<int>&);
        void getLowIndices(double,const Mat&,std::vector<unsigned int>&);
        void getHighIndices(double,const Mat&,std::vector<unsigned int>&);
        void getZeroIndices(const Mat&,std::vector<unsigned int>&);
        void seriesImplementation(double,const Mat&,
                                  const Mat&,const Mat&,Mat&);
        void continuedFractionImplementation(double,const Mat&,int,
                                             const Mat&,Mat&);
        void cumulativeSumsOfLogs(const std::vector<double>&,unsigned int,Mat&);
        void lowerIncompleteGamma(const double,const Mat&,const Mat&,const Mat&,Mat&);
        void ligfMultiple(const std::vector<double>&,const Mat&,const Mat&,Mat&);
        void ligfOfLimits(const std::vector<double>&,const Mat&,const Mat&,Mat&,Mat&);
        void computeDirectIntegral(unsigned short,const Mat&,const Mat&,const Mat&,const Mat&,Mat&);
        void computeRecursiveIntegrals(unsigned short,const Mat&,const Mat&,const Mat&,Mat&);
    public:
        void computeIntegrals(unsigned short,const Mat&,const Mat&,Mat&);
};


void polynomialbf::IntegralCalculator::binomialCoefficients(unsigned short n, std::vector<int>& B)
{
    B.resize(n+1);
    B[0] = 1;
    for(int r=1; r<=n; r++) { B[r] = B[r-1]*((int)n-r+1)/r; }
}


void polynomialbf::IntegralCalculator::cumulativeSumsOfLogs(const std::vector<double>& A, unsigned int N, Mat& out)
{
        int r,n;
        out = Mat::zeros(A.size(),N+1,CV_64FC1);
        for(r=0; r<A.size(); r++) { out.row(r) += A[r]; } // Initialize rth row to A[r]
        for(n=0; n<N+1; n++) { out.col(n) += n; } // Initialize nth column to A[r]+n
        log(out,out);
        for(n=1; n<N+1; n++) { out.col(n) += out.col(n-1); }
}


void polynomialbf::IntegralCalculator::getLowIndices(double a, const Mat& X, std::vector<unsigned int>& lowInd)
{
    lowInd.resize(X.cols);
    unsigned int N=0;
    const double* X_ptr = X.ptr<double>(0);
    for(unsigned int k=0; k<X.cols; k++)
    {
        if(X_ptr[k]!=0 && X_ptr[k]<=a+1)
        {
            lowInd[N] = k;
            N += 1;
        }
    }
    lowInd.resize(N);
}


void polynomialbf::IntegralCalculator::getHighIndices(double a, const Mat& X, std::vector<unsigned int>& highInd)
{
    highInd.resize(X.cols);
    unsigned int N=0;
    const double* X_ptr = X.ptr<double>(0);
    for(unsigned int k=0; k<X.cols; k++)
    {
        if(X_ptr[k]>a+1)
        {
            highInd[N] = k;
            N += 1;
        }
    }
    highInd.resize(N);
}


void polynomialbf::IntegralCalculator::getZeroIndices(const Mat& X, std::vector<unsigned int>& zeroInd)
{
    zeroInd.resize(X.cols);
    unsigned int N=0;
    const double* X_ptr = X.ptr<double>(0);
    for(unsigned int k=0; k<X.cols; k++)
    {
        if(X_ptr[k]==0)
        {
            zeroInd[N] = k;
            N += 1;
        }
    }
    zeroInd.resize(N);
}


void polynomialbf::IntegralCalculator::seriesImplementation(double a, const Mat& X,
                                              const Mat& sumlogaplusn, const Mat& logX, Mat& out)
{
    int n;
    const double* sumlog_ptr = sumlogaplusn.ptr<double>(0);
    Mat partialsum = Mat::zeros(X.rows,X.cols,CV_64FC1);
    Mat temp;
    for(n=0; n<sumlogaplusn.cols; n++)
    {
        temp = n*logX - sumlog_ptr[n];
        cv::exp(temp,temp);
        partialsum = partialsum + temp;
    }
    Mat logpartialsum;
    cv::log(partialsum,logpartialsum);
    temp = a*logX + logpartialsum - X;
    cv::exp(temp,out);
}


void polynomialbf::IntegralCalculator::continuedFractionImplementation(double a, const Mat& X, int N,
                                                          const Mat& logX, Mat& out)
{
    int n;
    double gamma_a = tgamma(a);
    Mat partialfrac = X + 2*N + 1 - a;
    for(n=N; n>0; n--)
    {
        partialfrac = X+2.0*n-1.0-a  - n*((double)n-a)/partialfrac;
    }
    Mat logpartialfrac;
    cv::log(partialfrac,logpartialfrac);
    Mat temp = a*logX - X - logpartialfrac;
    cv::exp(temp,temp);
    out = gamma_a - temp;
}


void polynomialbf::IntegralCalculator::lowerIncompleteGamma(const double a, const Mat& X, const Mat& sumlogaplusn, const Mat& logX, Mat& out)
{
    int N = sumlogaplusn.cols - 1;
    if(sumlogaplusn.rows==1 && out.rows==1 && out.cols==X.cols)
    {
        out = Mat::zeros(1,out.cols,CV_64FC1);
        Mat lowMask,highMask,nonZeroMask;
        this->li.getMask(X,"!=",0.0,nonZeroMask);
        this->li.getMask(X,"<=",a+1,lowMask);
        lowMask = lowMask & nonZeroMask;        // Discard zero values
        this->li.getMask(X,">",a+1,highMask);
        std::vector<unsigned int> zeroInd;
        this->getZeroIndices(X,zeroInd);
        int Nlow = countNonZero(lowMask);
        int Nhigh = countNonZero(highMask);

        Mat Xlow,Xhigh,logXlow,logXhigh;
        this->li.getMaskedValues(X,lowMask,Xlow);
        this->li.getMaskedValues(X,highMask,Xhigh);
        this->li.getMaskedValues(logX,lowMask,logXlow);
        this->li.getMaskedValues(logX,highMask,logXhigh);

        Mat outlow(1,Nlow,CV_64FC1);
        Mat outhigh(1,Nhigh,CV_64FC1);
        this->seriesImplementation(a,Xlow,sumlogaplusn,logXlow,outlow); // LIGF for X<=a+1
        this->continuedFractionImplementation(a,Xhigh,N_FRACTION,logXhigh,outhigh);  //LIGF for x>a+1

        this->li.setMaskedValues(out,lowMask,outlow);
        this->li.setMaskedValues(out,highMask,outhigh);
    }
    else { throw std::invalid_argument("Size mismatch"); }
}


void polynomialbf::IntegralCalculator::ligfMultiple(const std::vector<double>& A, const Mat& X, const Mat& sumlogAplusn, Mat& out)
{
    if(X.rows==1 && out.rows==A.size() && out.cols==X.cols)
    {
        Mat out_row,sumlog_row;
        Mat logX(1,X.cols,CV_64FC1);
        log(X,logX);
        for(int k=0; k<A.size(); k++)
        {
            out_row = out.row(k);
            sumlog_row = sumlogAplusn.row(k);
            this->lowerIncompleteGamma(A[k],X,sumlog_row,logX,out_row);
        }
    }
    else if(X.rows>1) { throw std::invalid_argument("X must have a single row"); }
    else if(out.rows!=A.size() || out.cols!=X.cols) { throw std::invalid_argument("Output size mismatch"); }
}


void polynomialbf::IntegralCalculator::ligfOfLimits(const std::vector<double>& A, const Mat& u_lim, const Mat& l_lim, Mat& u_out, Mat& l_out)
{
    if(u_lim.rows==1 && l_lim.rows==1 && u_lim.cols==l_lim.cols
       && u_out.rows==A.size() && u_out.cols==u_lim.cols
       && l_out.rows==A.size() && l_out.cols==l_lim.cols)
    {
        Mat sumlogAplusn;
        this->cumulativeSumsOfLogs(A,N_SERIES,sumlogAplusn);
        this->ligfMultiple(A,u_lim,sumlogAplusn,u_out);
        this->ligfMultiple(A,l_lim,sumlogAplusn,l_out);
    }
    else if(u_lim.rows!=1 || l_lim.rows!=1)
    { throw std::invalid_argument("u_lim & l_lim must have a single row"); }
    else if(u_lim.cols!=l_lim.cols)
    { throw std::invalid_argument("u_lim & l_lim must have same number of columns"); }
    else if(u_out.rows!=A.size() || u_out.cols!=u_lim.cols)
    { throw std::invalid_argument("u_out size mismatch"); }
    else if(l_out.rows!=A.size() || l_out.cols!=l_lim.cols)
    { throw std::invalid_argument("l_out size mismatch"); }
}


void polynomialbf::IntegralCalculator::computeDirectIntegral(unsigned short K, const Mat& X0, const Mat& root_lambda, const Mat& gammaU, const Mat& gammaL, Mat& out)
{
    if(out.rows==1 && out.cols==root_lambda.cols &&
       root_lambda.cols==gammaU.cols && gammaU.cols==gammaL.cols)
    {
        Mat root_lambda_rplus1;
        Mat X0_Kminusr = Mat::ones(1,X0.cols,CV_64FC1);;
        cv::pow(root_lambda,K+1,root_lambda_rplus1);
        std::vector<int> K_choose_r;
        this->binomialCoefficients(K,K_choose_r);
        if(K%2==0) { out = (gammaU.row(K)+gammaL.row(K)) / root_lambda_rplus1; }
        else       { out = (gammaU.row(K)-gammaL.row(K)) / root_lambda_rplus1; }
        for(int r=K-1; r>=0; r--)
        {
            root_lambda_rplus1 = root_lambda_rplus1 / root_lambda;
            X0_Kminusr = X0_Kminusr.mul(X0);
            if(r%2==0)
            {
                out = out + K_choose_r[r] * X0_Kminusr.mul(gammaU.row(r)+gammaL.row(r)) / root_lambda_rplus1;
            }
            else
            {
                out = out + K_choose_r[r] * X0_Kminusr.mul(gammaU.row(r)-gammaL.row(r)) / root_lambda_rplus1;
            }
        }
        out = 0.5*out;
    }
    else { throw std::invalid_argument("Size mismatch"); }
}


void polynomialbf::IntegralCalculator::computeRecursiveIntegrals(unsigned short N, const Mat& X0, const Mat& lambda, const Mat& expU, Mat& out)
{
    for(int k=N; k>=2; k--) { out.row(k-2) = (2*lambda.mul(out.row(k)-X0.mul(out.row(k-1))) + expU) / (k-1); }
}


void polynomialbf::IntegralCalculator::computeIntegrals(unsigned short N, const Mat& lambda, const Mat& X0, Mat& out)
{
    out.create(N+1,X0.cols,CV_64FC1);
    Mat root_lambda,Ulim,Llim,expU;
    std::vector<double> A;
    sqrt(lambda,root_lambda);
    Ulim = lambda.mul(1.0-X0).mul(1.0-X0);
    Llim = lambda.mul(X0).mul(X0);
    exp(-1.0*Ulim,expU);
    A.reserve(N+1);
    for(int r=0; r<N+1; r++) { A.push_back(0.5*(r+1.0)); }

    Mat lgamma_Ulim(N+1,Ulim.cols,CV_64FC1);
    Mat lgamma_Llim(N+1,Llim.cols,CV_64FC1);
    this->ligfOfLimits(A,Ulim,Llim,lgamma_Ulim,lgamma_Llim);

    /* Compute integrals directly for k=N & k=N-1 */
    Mat outrow;
    outrow = out.row(N);
    this->computeDirectIntegral(N,X0,root_lambda,lgamma_Ulim,lgamma_Llim,outrow);
    if(N>0)
    {
        outrow = out.row(N-1);
        this->computeDirectIntegral(N-1,X0,root_lambda,lgamma_Ulim,lgamma_Llim,outrow);
    }

    /* Compute integrals recursively for k=N-2,...,0 */
    this->computeRecursiveIntegrals(N,X0,lambda,expU,out);
}


#endif  /* INTEGRALCALCULATOR_CPP */

