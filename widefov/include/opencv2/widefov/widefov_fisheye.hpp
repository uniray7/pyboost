#ifndef __OPENCV_WIDEFOV_FISHEYE_H__
#define __OPENCV_WIDEFOV_FISHEYE_H__

#include "opencv2/core.hpp"

namespace cv { namespace widefov { namespace internal {

struct CV_EXPORTS IntrinsicParams
{
    Vec2d f;
    Vec2d c;
    Vec4d k;
    double alpha;
    std::vector<int> isEstimate;

    IntrinsicParams();
    IntrinsicParams(Vec2d f, Vec2d c, Vec4d k, double alpha = 0);
    IntrinsicParams operator+(const Mat& a);
    IntrinsicParams& operator =(const Mat& a);
    void Init(const cv::Vec2d& f, const cv::Vec2d& c, const cv::Vec4d& k = Vec4d(0,0,0,0), const double& alpha = 0);
};

CV_EXPORTS void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
                   cv::InputArray _rvec,cv::InputArray _tvec,
                   const IntrinsicParams& param, cv::OutputArray jacobian);

CV_EXPORTS void ComputeExtrinsicRefine(const Mat& imagePoints, const Mat& objectPoints, Mat& rvec,
                            Mat&  tvec, Mat& J, const int MaxIter,
                            const IntrinsicParams& param, const double thresh_cond);
CV_EXPORTS Mat ComputeHomography(Mat m, Mat M); // same as OpenCV, but unable to link them (opencv hides them)

CV_EXPORTS Mat NormalizePixels(const Mat& imagePoints, const IntrinsicParams& param);

CV_EXPORTS void InitExtrinsics(const Mat& _imagePoints, const Mat& _objectPoints, const IntrinsicParams& param, Mat& omckk, Mat& Tckk);

CV_EXPORTS void CalibrateExtrinsics(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                         const IntrinsicParams& param, const int check_cond,
                         const double thresh_cond, InputOutputArray omc, InputOutputArray Tc);

CV_EXPORTS void ComputeJacobians(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                      const IntrinsicParams& param,  InputArray omc, InputArray Tc,
                      const int& check_cond, const double& thresh_cond, Mat& JJ2_inv, Mat& ex3, Mat& JJ3);

CV_EXPORTS void EstimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                           const IntrinsicParams& params, InputArray omc, InputArray Tc,
                           IntrinsicParams& errors, Vec2d& std_err, double thresh_cond, int check_cond, double& rms);

CV_EXPORTS void dAB(cv::InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB); // same as OpenCV

CV_EXPORTS void JRodriguesMatlab(const Mat& src, Mat& dst); // same as OpenCV

CV_EXPORTS void compose_motion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2,
                    Mat& om3, Mat& T3, Mat& dom3dom1, Mat& dom3dT1, Mat& dom3dom2,
                    Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2); // same as OpenCV

CV_EXPORTS double median(const Mat& row); // same as OpenCV

CV_EXPORTS Vec3d median3d(InputArray m); // same as OpenCV

struct JacobianRow
{
    Vec2d df, dc;
    Vec4d dk;
    Vec3d dom, dT;
    double dalpha;
};

void subMatrix(const Mat& src, Mat& dst, const std::vector<int>& cols, const std::vector<int>& rows);

}}} // cv::widefov::internal namespace scope end

#endif
