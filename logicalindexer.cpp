/*
 * =====================================================================================
 *
 *       Filename:  logicalindexer.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  Tuesday 08 August 2017 03:43:08  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Ruturaj Gavaskar (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef LOGICALINDEXER_CPP
#define LOGICALINDEXER_CPP

#include "opencv2/core.hpp"
#include<cstdint>
#include<stdexcept>
#include "polynomialbf.hpp"

#define TYPE_UNSIGNED_8_INT         1
#define TYPE_SIGNED_8_INT           2
#define TYPE_UNSIGNED_16_INT        3
#define TYPE_SIGNED_16_INT          4
#define TYPE_SIGNED_32_INT          5
#define TYPE_32_FLOAT               6
#define TYPE_64_DOUBLE              7
using namespace cv;

template <class mat_type> class polynomialbf::LogicalIndexer
{
    private:
        unsigned short getMatType(const Mat&);
    public:
        void getMask(const Mat&,const std::string&,const Mat&,Mat&);
        void getMask(const Mat&,const std::string&,mat_type,Mat&);
        void setMaskedValues(Mat&,const Mat&,mat_type,uint8_t maskval=1);
        void setMaskedValues(Mat&,const Mat&,const Mat&,uint8_t maskval=1);
        void setMaskedValues(Mat&,const Mat&,mat_type*,uint8_t maskval=1);
        void getMaskedValues(const Mat&,const Mat&,Mat&);
};

/* getMatType(Mat& A): Find data type of input Mat object A. */
template <class mat_type> unsigned short polynomialbf::LogicalIndexer<mat_type>::getMatType(const Mat& A)
{
    int Atype;
    unsigned short out=0;
    Atype = A.type();
    unsigned char depth = Atype & CV_MAT_DEPTH_MASK;
    switch(depth)
    {
        case CV_8U:  out = TYPE_UNSIGNED_8_INT;   break;
        case CV_8S:  out = TYPE_SIGNED_8_INT;     break;
        case CV_16U: out = TYPE_UNSIGNED_16_INT;  break;
        case CV_16S: out = TYPE_SIGNED_16_INT;    break;
        case CV_32S: out = TYPE_SIGNED_32_INT;    break;
        case CV_32F: out = TYPE_32_FLOAT;         break;
        case CV_64F: out = TYPE_64_DOUBLE;        break;
    }
    return (out);
}


/* getMask(Mat& A, std::string& rel, Mat& B): Find mask based on
 * elementwise binary relation between Mat objects A and B.
 * A = Mat object
 * rel = binary relation, "<",">","==","!="
 * B = Mat object
 * mask = Raw binary mask
 */
template <class mat_type> void polynomialbf::LogicalIndexer<mat_type>::getMask(const Mat& A, const std::string& rel, const Mat& B, Mat& mask)
{
    mask = Mat::zeros(A.rows,A.cols,CV_8UC1);
    uint8_t* maskrow;
    const mat_type* Arow;
    const mat_type* Brow;
    int c;

    if(rel == ">")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            Brow = B.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]>Brow[c]) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "<")
    {
        for(int r=0; r<A.rows; r++)
        {
             Arow = A.ptr<mat_type>(r);
             Brow = B.ptr<mat_type>(r);
             maskrow = mask.ptr<uint8_t>(r);
             for(c=0; c<A.cols; c++) { if(Arow[c]<Brow[c]) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == ">=")
    {
        for(int r=0; r<A.rows; r++)
        {
             Arow = A.ptr<mat_type>(r);
             Brow = B.ptr<mat_type>(r);
             maskrow = mask.ptr<uint8_t>(r);
             for(c=0; c<A.cols; c++) { if(Arow[c]>=Brow[c]) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "<=")
    {
        for(int r=0; r<A.rows; r++)
        {
             Arow = A.ptr<mat_type>(r);
             Brow = B.ptr<mat_type>(r);
             maskrow = mask.ptr<uint8_t>(r);
             for(c=0; c<A.cols; c++) { if(Arow[c]<=Brow[c]) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "==")
    {
        for(int r=0; r<A.rows; r++)
        {
             Arow = A.ptr<mat_type>(r);
             Brow = B.ptr<mat_type>(r);
             maskrow = mask.ptr<uint8_t>(r);
             for(c=0; c<A.cols; c++) { if(Arow[c]==Brow[c]) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "!=")
    {
        for(int r=0; r<A.rows; r++)
        {
             Arow = A.ptr<mat_type>(r);
             Brow = B.ptr<mat_type>(r);
             maskrow = mask.ptr<uint8_t>(r);
             for(c=0; c<A.cols; c++) { if(Arow[c]!=Brow[c]) {maskrow[c] = 0x01;} }
        }
    }
}


/* getMask(Mat& A, std::string& rel, Mat& B): Find mask based on
 * elementwise binary relation between Mat objects A and B.
 * A = Mat object
 * rel = binary relation, "<",">","==","!="
 * B = Mat object
 * mask = Raw binary mask
 */
template <class mat_type> void polynomialbf::LogicalIndexer<mat_type>::getMask(const Mat& A, const std::string& rel, mat_type val, Mat& mask)
{
    mask = Mat::zeros(A.rows,A.cols,CV_8UC1);
    uint8_t* maskrow;
    const mat_type* Arow;
    int c;

    if(rel == ">")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]>val) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "<")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]<val) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == ">=")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]>=val) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "<=")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]<=val) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "==")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]==val) {maskrow[c] = 0x01;} }
        }
    }
    else if(rel == "!=")
    {
        for(int r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(Arow[c]!=val) {maskrow[c] = 0x01;} }
        }
    }
}


template <class mat_type> void polynomialbf::LogicalIndexer<mat_type>::setMaskedValues(Mat& A, const Mat& mask, mat_type val, uint8_t maskval)
{
    mat_type* Arow;
    const uint8_t* maskrow;
    int r,c;
    for(r=0; r<A.rows; r++)
    {
        Arow = A.ptr<mat_type>(r);
        maskrow = mask.ptr<uint8_t>(r);
        for(c=0; c<A.cols; c++) { if(maskrow[c]==maskval) { Arow[c] = val; } }
    }
}


template <class mat_type> void polynomialbf::LogicalIndexer<mat_type>::setMaskedValues(Mat& A, const Mat& mask, const Mat& B, uint8_t maskval)
{
    mat_type* Arow;
    const uint8_t* maskrow;
    int r, c;
    if(B.rows==1)
    {
        int Bind = 0;
        const mat_type* Bdata = B.ptr<mat_type>(0);
        for(r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(maskrow[c]==maskval) { Arow[c] = Bdata[Bind]; Bind += 1; } }
        }
    }
    else if(B.rows==A.rows && B.cols==A.cols)
    {
        const mat_type* Brow;
        for(r=0; r<A.rows; r++)
        {
            Arow = A.ptr<mat_type>(r);
            Brow = B.ptr<mat_type>(r);
            maskrow = mask.ptr<uint8_t>(r);
            for(c=0; c<A.cols; c++) { if(maskrow[c]==maskval) { Arow[c] = Brow[c]; } }
        }
    }
    else if(B.rows!=0 && B.cols!=0){ throw std::invalid_argument("Size mismatch"); }
}


template <class mat_type> void polynomialbf::LogicalIndexer<mat_type>::setMaskedValues(Mat& A, const Mat& mask, mat_type* data, uint8_t maskval)
{
    mat_type* Arow;
    const uint8_t* maskrow;
    int r, c, ind=0;
    for(r=0; r<A.rows; r++)
    {
        Arow = A.ptr<mat_type>(r);
        maskrow = mask.ptr<uint8_t>(r);
        for(c=0; c<A.cols; c++) { if(maskrow[c]==maskval) { Arow[c] = data[ind]; ind += 1; } }
    }
}


template <class mat_type> void polynomialbf::LogicalIndexer<mat_type>::getMaskedValues(const Mat& A, const Mat& mask, Mat& out)
{
    const mat_type* Arow;
    const uint8_t* maskrow;
    int r, c, n, N;
    N = countNonZero(mask);
    unsigned short dtype = this->getMatType(A);
    int outtype;
    switch(dtype)
    {
        case TYPE_UNSIGNED_8_INT:
            outtype = CV_8UC1;  break;
        case TYPE_SIGNED_8_INT:
            outtype = CV_8SC1;  break;
        case TYPE_UNSIGNED_16_INT:
            outtype = CV_16UC1; break;
        case TYPE_SIGNED_16_INT:
            outtype = CV_16SC1; break;
        case TYPE_SIGNED_32_INT:
            outtype = CV_32SC1; break;
        case TYPE_32_FLOAT:
            outtype = CV_32FC1; break;
        case TYPE_64_DOUBLE:
            outtype = CV_64FC1; break;
    }
    out.create(1,N,outtype);
    mat_type* outptr = out.ptr<mat_type>(0);
    for(r=0,n=0; r<A.rows && n<N; r++)
    {
        Arow = A.ptr<mat_type>(r);
        maskrow = mask.ptr<uint8_t>(r);
        for(c=0; c<A.cols; c++) { if(maskrow[c]) { outptr[n] = Arow[c]; n += 1; } }
    }
}

#endif /* LOGICALINDEXING_CPP */

