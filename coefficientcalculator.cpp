/*
 * =====================================================================================
 *
 *       Filename:  coefficientcalculator.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Monday 14 August 2017 10:24:29  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef COEFFICIENTCALCULATOR_CPP
#define COEFFICIENTCALCULATOR_CPP

#include<cmath>
#include<stdexcept>
#include "opencv2/core.hpp"
#include "polynomialbf.hpp"

class polynomialbf::CoefficientCalculator
{
    public:
//    private:
        polynomialbf::LogicalIndexer<double> li;
        std::vector<double> bf;
        std::vector<double> bb;
        double B;
        void getZeroIndices(const Mat&,std::vector<unsigned int>&);
        void invHilbertMatrix(int,Mat&);
        void applySymmetricPadding(const Mat&,int,Mat&);
        void boxFilter1D(double*,unsigned int,unsigned int,unsigned int);
        void boxFilter2D(const Mat&,int,Mat&);
        void computeGaussianParameters(double);
        void gaussianFilter1D(double*,unsigned int,unsigned int,unsigned int);
        void gaussianFilter2D(const Mat&,int,Mat&);
        void filterPowers(const Mat&,const std::string&,double,int,Mat&);
        void computeMoments(const Mat&,const Mat&,const Mat&,Mat&);
//    public:
        void computeCoefficients(const Mat&,const Mat&,const Mat&, const std::string&,
                                 double,int,Mat&);
};


using namespace cv;

void polynomialbf::CoefficientCalculator::getZeroIndices(const Mat& X, std::vector<unsigned int>& zeroInd)
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


void polynomialbf::CoefficientCalculator::invHilbertMatrix(int N, Mat& out)
{
    out.create(N,N,CV_64FC1);
    double* outptr = out.ptr<double>(0);
    int i,j;
    double r;
    double p = (double)N;
    for(i=1; i<=N; i++)
    {
        r = p*p;
        outptr[(i-1)*(N+1)] = r/(2*i-1.0);
        for(j=i+1; j<=N; j++)
        {
            r = -(N-j+1.0)*r*(N+j-1.0)/((j-1.0)*(j-1.0));
            outptr[(i-1)*N+(j-1)] = r/(i+j-1.0);
            outptr[(j-1)*N+(i-1)] = outptr[(i-1)*N+(j-1)];
        }
        p = ((double)N-i)*p*(N+i)/(i*i);
    }
}


void polynomialbf::CoefficientCalculator::applySymmetricPadding(const Mat& in, int padsize, Mat& out)
{
    out.create(in.rows+2*padsize,in.cols+2*padsize,CV_64FC1);
    Mat center = out.rowRange(padsize,in.rows+padsize).colRange(padsize,in.cols+padsize);
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


void polynomialbf::CoefficientCalculator::boxFilter1D(double* start, unsigned int stepsize, unsigned int radius, unsigned int datasize)
{
    std::vector<double> out;
    out.resize(datasize,0.0);
    int i;
    for(i=0; i<2*radius+1; i++)
    {
        out[radius] += *(start+(i*stepsize));
    }
    for(i=radius+1; i<datasize-radius; i++)
    {
        out[i] = out[i-1] + *(start+(radius+i)*stepsize) - *(start+(i-1-radius)*stepsize);
    }
    for(i=0; i<datasize; i++)
    {
        *(start+i*stepsize) = out[i];
    }
}


void polynomialbf::CoefficientCalculator::boxFilter2D(const Mat& in, int radius, Mat& out)
{
    if(out.rows==in.rows && out.cols==in.cols)
    {
        Mat padded_in(in.rows+2*radius,in.cols+2*radius,CV_64FC1);
        this->applySymmetricPadding(in,radius,padded_in);

        // Filter each row
        double* rowptr;
        for(int i=0; i<padded_in.rows; i++)
        {
            rowptr = padded_in.ptr<double>(i);
            this->boxFilter1D(rowptr,1,radius,padded_in.cols);
        }

        // Filter each column
        double* colptr;
        for(int j=0; j<padded_in.cols; j++)
        {
            colptr = padded_in.ptr<double>(0) + j;
            this->boxFilter1D(colptr,padded_in.cols,radius,padded_in.rows);
        }

        // Copy central region of convolved image to output & normalize
        out = padded_in.rowRange(radius,in.rows+radius).colRange(radius,in.cols+radius) / ((2*radius+1)*(2*radius+1));
    }
    else { throw std::invalid_argument("Size mismatch"); }
}


void polynomialbf::CoefficientCalculator::computeGaussianParameters(double sigma)
{
    double q;
    if (sigma < 2.5) { q = 3.97156 - 4.14554*sqrt(1-0.26891*sigma); }
    else             { q = 0.98711*sigma - 0.9633; }

    double b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    double b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
    double b2 = -(1.4281*q*q + 1.26661*q*q*q);
    double b3 = 0.422205*q*q*q;

    this->bf.resize(3);
    this->bf[0] = b3/b0;
    this->bf[1] = b2/b0;
    this->bf[2] = b1/b0;

    this->bb.resize(3);
    this->bb[0] = b1/b0;
    this->bb[1] = b2/b0;
    this->bb[2] = b3/b0;

    this->B = 1 - (b1+b2+b3)/b0;
}


void polynomialbf::CoefficientCalculator::gaussianFilter1D(double* start, unsigned int stepsize, unsigned int radius, unsigned int datasize)
{
    std::vector<double> out;
    out.resize(datasize);
    int i;

    out[0] =   this->B*(*start);
    out[1] =   this->B*(*(start+stepsize))
             + this->bf[2]*out[0];
    out[2] =   this->B*(*(start+2*stepsize))
             + this->bf[1]*out[0]
             + this->bf[2]*out[1];
    for(i=3; i<datasize; i++)
    {
        out[i] =   this->B*(*(start+i*stepsize))
                 + this->bf[0]*out[i-3]
                 + this->bf[1]*out[i-2]
                 + this->bf[2]*out[i-1];
    }

    *(start+(datasize-1)*stepsize) =   this->B*out[datasize-1];
    *(start+(datasize-2)*stepsize) =   this->B*out[datasize-2]
                                     + this->bb[0]*(*(start+(datasize-1)*stepsize));
    *(start+(datasize-3)*stepsize) =   this->B*out[datasize-3]
                                     + this->bb[1]*(*(start+(datasize-2)*stepsize))
                                     + this->bb[1]*(*(start+(datasize-1)*stepsize));
    for(i=datasize-4; i>=radius; i--)
    {
        *(start+i*stepsize) =   this->B*out[i]
                              + this->bb[0]*(*(start+(i+1)*stepsize))
                              + this->bb[1]*(*(start+(i+2)*stepsize))
                              + this->bb[2]*(*(start+(i+3)*stepsize));
    }
}


void polynomialbf::CoefficientCalculator::gaussianFilter2D(const Mat& in, int radius, Mat& out)
{
    if(out.rows==in.rows && out.cols==in.cols)
    {
        Mat padded_in(in.rows+2*radius,in.cols+2*radius,CV_64FC1);
        this->applySymmetricPadding(in,radius,padded_in);

        // Filter each row
        double* rowptr;
        for(int i=0; i<padded_in.rows; i++)
        {
            rowptr = padded_in.ptr<double>(i);
            this->gaussianFilter1D(rowptr,1,radius,padded_in.cols);
        }

        // Filter each column
        double* colptr;
        for(int j=radius; j<padded_in.cols; j++)
        {
            colptr = padded_in.ptr<double>(0) + j;
            this->gaussianFilter1D(colptr,padded_in.cols,radius,padded_in.rows);
        }

        // Copy central region of convolved image to output
        out = padded_in.rowRange(radius,in.rows+radius).colRange(radius,in.cols+radius) + 0.0;
    }
    else { throw std::invalid_argument("Size mismatch"); }
}


void polynomialbf::CoefficientCalculator::filterPowers(const Mat& in, const std::string& kernel, double sigma, int N, Mat& out)
{
    int radius;
    Mat in_power;
    if(kernel == "box")
    {
        out.create(N+1,in.rows*in.cols,CV_64FC1);
        radius = static_cast<int>(sigma+0.5);
        out.row(0) = Mat::ones(1,in.rows*in.cols,CV_64FC1);
        in_power.create(in.rows,in.cols,CV_64FC1);
        for(int k=1; k<=N; k++)
        {
            cv::pow(in,k,in_power);
            Mat outrow(in.rows,in.cols,CV_64FC1,out.ptr<double>(k));
            this->boxFilter2D(in_power,radius,outrow);
        }
    }
    else if(kernel == "gaussian")
    {
        out.create(N+1,in.rows*in.cols,CV_64FC1);
        radius = static_cast<int>(6*sigma+0.5);
        radius = (radius%2==0) ? (radius/2) : (radius-1)/2;
        this->computeGaussianParameters(sigma);
        out.row(0) = Mat::ones(1,in.rows*in.cols,CV_64FC1);
        in_power.create(in.rows,in.cols,CV_64FC1);
        for(int k=1; k<=N; k++)
        {
            cv::pow(in,k,in_power);
            Mat outrow(in.rows,in.cols,CV_64FC1,out.ptr<double>(k));
            this->gaussianFilter2D(in_power,radius,outrow);
        }
    }
    else { throw std::invalid_argument("Invalid kernel type"); }
}


void polynomialbf::CoefficientCalculator::computeMoments(const Mat& fsmooth, const Mat& alpha, const Mat& beta, Mat& out)
{
    // Extract pixels for which computation is required
    Mat Alpha,Beta,mask;
    this->li.getMask(alpha,"!=",beta,mask);
    this->li.getMaskedValues(alpha,mask,Alpha);
    this->li.getMaskedValues(beta,mask,Beta);
    out.create(fsmooth.rows,Alpha.cols,CV_64FC1);
    Mat fsmooth_mask(fsmooth.rows,Alpha.cols,CV_64FC1);
    Mat fsmooth_row,fsmooth_mask_row;;
    for(int k=0; k<fsmooth.rows; k++)
    {
        fsmooth_row = fsmooth.row(k);
        fsmooth_mask_row = fsmooth_mask.row(k);
        this->li.getMaskedValues(fsmooth_row,mask,fsmooth_mask_row);
    }

    Mat term_rminus1,term_r;
    double* outptr;
    double* fsmoothptr;
    double* Betaptr;
    int r,z;
    std::vector<unsigned int> zeroInd;
    this->getZeroIndices(Alpha,zeroInd);

    Betaptr = Beta.ptr<double>(0);
    out.row(0) = Mat::ones(1,out.cols,CV_64FC1);
    for(int k=0; k<fsmooth_mask.rows; k++)
    {
        term_rminus1 = (-Alpha/(Beta-Alpha));
        pow(term_rminus1,k,term_rminus1);
        out.row(k) = term_rminus1 + 0.0;    // Alternatively use term_rminus1.clone()
        for(r=1; r<=k; r++)
        {
            term_r = ((r-k-1)/(double)r) * fsmooth_mask.row(r).mul(term_rminus1) / (Alpha.mul(fsmooth_mask.row(r-1)));
            out.row(k) = out.row(k) + term_r;
            term_rminus1 = term_r + 0.0;
        }
        outptr = out.ptr<double>(k);
        fsmoothptr = fsmooth_mask.ptr<double>(k);
        for(z=0; z<zeroInd.size(); z++) { outptr[zeroInd[z]] = fsmoothptr[zeroInd[z]]/pow(Betaptr[zeroInd[z]],k) ; }
    }
}


void polynomialbf::CoefficientCalculator::computeCoefficients(const Mat& inimg, const Mat& Alpha, const Mat& Beta, const std::string& kernel,
                                                double sigma_s, int N, Mat& out)
{
    Mat fsmooth(N+1,inimg.rows*inimg.cols,CV_64FC1);
    this->filterPowers(inimg,kernel,sigma_s,N,fsmooth);

    Mat moments(N+1,inimg.rows*inimg.cols,CV_64FC1);
    this->computeMoments(fsmooth,Alpha,Beta,moments);

    Mat H_inv;
    this->invHilbertMatrix(N+1,H_inv);

    out = H_inv*moments;
}

#endif /* COEFFICIENTCALCULATOR_CPP */

