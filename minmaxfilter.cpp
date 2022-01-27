/*
 * =====================================================================================
 *
 *       Filename:  minmaxfilter.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Thursday 17 August 2017 11:55:28  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef MINMAXFILTER_CPP
#define MINMAXFILTER_CPP

#include "opencv2/core.hpp"
#include "polynomialbf.hpp"

using namespace cv;

class polynomialbf::MinMaxFilter
{
    public:
        void applyPostZeroPadding(const Mat&,int,int,Mat&);
        void applyPostReplicatePadding(const Mat&,int,int,Mat&);
        void maxFilter(const Mat&,int,Mat&);
        void minFilter(const Mat&,int,Mat&);
};

void polynomialbf::MinMaxFilter::applyPostZeroPadding(const Mat& in, int rowpad, int colpad,  Mat& out)
{
    out = Mat::zeros(in.rows+rowpad,in.cols+colpad,CV_64FC1);
    Mat orig = out.rowRange(0,in.rows).colRange(0,in.cols);
    orig = in + 0.0;
}


void polynomialbf::MinMaxFilter::applyPostReplicatePadding(const Mat& in, int rowpad, int colpad, Mat& out)
{
    out.create(in.rows+rowpad,in.cols+colpad,CV_64FC1);

    // Set main block
    in.copyTo(out.rowRange(0,in.rows).colRange(0,in.cols));

    // Set value of bottom right block
    if(rowpad!=0 && colpad!=0)
    {
        out.rowRange(in.rows,out.rows).colRange(in.cols,out.cols) = in.at<double>(in.rows-1,in.cols-1);
    }

    // Replicate rows
    double* src_row = out.ptr<double>(in.rows-1);
    for(int r=in.rows; r<out.rows; r++)
    {
        std::memcpy(out.ptr<double>(r),src_row,in.cols*sizeof(double));
    }

    // Replicate columns
    Mat dest_col;
    for(int c=in.cols; c<out.cols; c++)
    {
        dest_col = out.col(c).rowRange(0,in.rows);
        in.col(in.cols-1).copyTo(dest_col);
    }
}


void polynomialbf::MinMaxFilter::maxFilter(const Mat& in, int winsize, Mat& out)
{
    if(winsize%2==1)
    {
        out.create(in.rows,in.cols,CV_64FC1);
        int rowpad = (in.rows/winsize + (in.rows%winsize != 0))*winsize - in.rows;
        int colpad = (in.cols/winsize + (in.cols%winsize != 0))*winsize - in.cols;
        Mat templ;
        this->applyPostZeroPadding(in,rowpad,colpad,templ);
        std::vector<double> L,R;
        double* templrow;
        double* templcol;
        double* outcol;
        int k,p,q;
        int sym = (winsize-1)/2;
        double r,l;

        // Scan along rows
        L.resize(templ.cols);
        R.resize(templ.cols);
        for(int ii=1; ii<=in.rows; ii++)
        {
            templrow = templ.ptr<double>(ii-1);
            std::fill(L.begin(),L.end(),0.0);
            std::fill(R.begin(),R.end(),0.0);
            L[0] = templrow[0];
            R[templ.cols-1] = templrow[templ.cols-1];
            for(k=2; k<=templ.cols; k++)
            {
                if((k-1)%winsize==0)
                {
                    L[k-1] = templrow[k-1];
                    R[templ.cols-k] = templrow[templ.cols-k];
                }
                else
                {
                    L[k-1] = (L[k-2] > templrow[k-1]) ? L[k-2] : templrow[k-1];
                    R[templ.cols-k] = (R[templ.cols-k+1]>templrow[templ.cols-k]) ? R[templ.cols-k+1] : templrow[templ.cols-k];
                }
            }
            for(k=1; k<=templ.cols; k++)
            {
                p = k - sym;
                q = k + sym;
                r = (p<1) ? -1 : R[p-1];
                l = (q>templ.cols) ? -1 : L[q-1];
                templrow[k-1] = (r>l) ? r : l;
            }
        }

        // Scan along columns
        L.resize(templ.rows);
        R.resize(templ.rows);
        int stepsize = templ.cols;
        for(int jj=1; jj<=in.cols; jj++)
        {
            templcol = templ.ptr<double>(0) + jj-1;
            outcol = out.ptr<double>(0) + jj-1;
            std::fill(L.begin(),L.end(),0.0);
            std::fill(R.begin(),R.end(),0.0);
            L[0] = *templcol;
            R[templ.rows-1] = *(templcol+(templ.rows-1)*stepsize);
            for(k=2; k<=templ.rows; k++)
            {
                if((k-1)%winsize==0)
                {
                    L[k-1] = *(templcol+(k-1)*stepsize);
                    R[templ.rows-k] = *(templcol+(templ.rows-k)*stepsize);
                }
                else
                {
                    L[k-1] = (L[k-2]>*(templcol+(k-1)*stepsize)) ? L[k-2] : *(templcol+(k-1)*stepsize);
                    R[templ.rows-k] = (R[templ.rows-k+1]> *(templcol+(templ.rows-k)*stepsize)) ? R[templ.rows-k+1] : *(templcol+(templ.rows-k)*stepsize);
                }
            }
            for(k=1; k<=templ.rows; k++)
            {
                p = k - sym;
                q = k + sym;
                r = (p<1) ? -1 : R[p-1];
                l = (q>templ.rows) ? -1 : L[q-1];

                if(k<=in.rows)
                {
//                    *(outcol+(k-1)*out.cols) = (r>l) ? r : l;
                    out.at<double>(k-1,jj-1) = (r>l) ? r : l;
                }
            }
        }
    }
    else { throw std::invalid_argument("winsize must be odd"); }
}


void polynomialbf::MinMaxFilter::minFilter(const Mat& in, int winsize, Mat& out)
{
    if(winsize%2==1)
    {
        out.create(in.rows,in.cols,CV_64FC1);
        int rowpad = (in.rows/winsize + (in.rows%winsize != 0))*winsize - in.rows;
        int colpad = (in.cols/winsize + (in.cols%winsize != 0))*winsize - in.cols;
        Mat templ;
        this->applyPostReplicatePadding(in,rowpad,colpad,templ);
        std::vector<double> L,R;
        double* templrow;
        double* templcol;
        double* outcol;
        int k,p,q;
        int sym = (winsize-1)/2;
        double r,l;

        // Scan along rows
        L.resize(templ.cols);
        R.resize(templ.cols);
        for(int ii=1; ii<=in.rows; ii++)
        {
            templrow = templ.ptr<double>(ii-1);
            std::fill(L.begin(),L.end(),0.0);
            std::fill(R.begin(),R.end(),0.0);
            L[0] = templrow[0];
            R[templ.cols-1] = templrow[templ.cols-1];
            for(k=2; k<=templ.cols; k++)
            {
                if((k-1)%winsize==0)
                {
                    L[k-1] = templrow[k-1];
                    R[templ.cols-k] = templrow[templ.cols-k];
                }
                else
                {
                    L[k-1] = (L[k-2] < templrow[k-1]) ? L[k-2] : templrow[k-1];
                    R[templ.cols-k] = (R[templ.cols-k+1]<templrow[templ.cols-k]) ? R[templ.cols-k+1] : templrow[templ.cols-k];
                }
            }
            for(k=1; k<=templ.cols; k++)
            {
                p = k - sym;
                q = k + sym;
                r = (p<1) ? std::numeric_limits<double>::infinity() : R[p-1];
                l = (q>templ.cols) ? std::numeric_limits<double>::infinity() : L[q-1];
                templrow[k-1] = (r<l) ? r : l;
            }
        }

        // Scan along columns
        L.resize(templ.rows);
        R.resize(templ.rows);
        int stepsize = templ.cols;
        for(int jj=1; jj<=in.cols; jj++)
        {
            templcol = templ.ptr<double>(0) + jj-1;
            outcol = out.ptr<double>(0) + jj-1;
            std::fill(L.begin(),L.end(),0.0);
            std::fill(R.begin(),R.end(),0.0);
            L[0] = *templcol;
            R[templ.rows-1] = *(templcol+(templ.rows-1)*stepsize);
            for(k=2; k<=templ.rows; k++)
            {
                if((k-1)%winsize==0)
                {
                    L[k-1] = *(templcol+(k-1)*stepsize);
                    R[templ.rows-k] = *(templcol+(templ.rows-k)*stepsize);
                }
                else
                {
                    L[k-1] = (L[k-2]<*(templcol+(k-1)*stepsize)) ? L[k-2] : *(templcol+(k-1)*stepsize);
                    R[templ.rows-k] = (R[templ.rows-k+1]< *(templcol+(templ.rows-k)*stepsize)) ? R[templ.rows-k+1] : *(templcol+(templ.rows-k)*stepsize);
                }
            }
            for(k=1; k<=templ.rows; k++)
            {
                p = k - sym;
                q = k + sym;
                r = (p<1) ? std::numeric_limits<double>::infinity() : R[p-1];
                l = (q>templ.rows) ? std::numeric_limits<double>::infinity() : L[q-1];

                if(k<=in.rows)
                {
                    *(outcol+(k-1)*out.cols) = (r<l) ? r : l;
                }
            }
        }
    }
    else { throw std::invalid_argument("winsize must be odd"); }
}

#endif /* MINMAXFILTER_CPP */

