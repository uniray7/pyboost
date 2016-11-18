#include "precomp.hpp"

using namespace std;

namespace cv { namespace widefov {

double RADIUS_OFFSET = 0;
double multiChkBoardStereoCalibrate(
    InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
    InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2,
    InputOutputArrayOfArrays Rckk1, InputOutputArrayOfArrays Tckk1, InputOutputArrayOfArrays Rckk2, InputOutputArrayOfArrays Tckk2,
    OutputArray R, OutputArray T, int flags, TermCriteria criteria)
{
    CV_Assert(!objectPoints.empty() && !imagePoints1.empty() && !imagePoints2.empty());
    CV_Assert(objectPoints.total() == imagePoints1.total() || imagePoints1.total() == imagePoints2.total());
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    CV_Assert(imagePoints1.type() == CV_32FC2 || imagePoints1.type() == CV_64FC2);
    CV_Assert(imagePoints2.type() == CV_32FC2 || imagePoints2.type() == CV_64FC2);

    CV_Assert((!K1.empty() && K1.size() == Size(3,3)) || K1.empty());
    CV_Assert((!D1.empty() && D1.total() == 4) || D1.empty());
    CV_Assert((!K2.empty() && K1.size() == Size(3,3)) || K2.empty());
    CV_Assert((!D2.empty() && D1.total() == 4) || D2.empty());

    CV_Assert(((flags & CALIB_FIX_INTRINSIC) && !K1.empty() && !K2.empty() && !D1.empty() && !D2.empty()) || !(flags & CALIB_FIX_INTRINSIC));

    //-------------------------------Initialization

    const int threshold = 50;
    const double thresh_cond = 1e6;
    const int check_cond = 1;

    int n_points = (int)objectPoints.getMat(0).total();
    int n_images = (int)objectPoints.total();
    double change = 1;

    internal::IntrinsicParams intrinsicLeft;
    internal::IntrinsicParams intrinsicRight;

    internal::IntrinsicParams intrinsicLeft_errors;
    internal::IntrinsicParams intrinsicRight_errors;

    Matx33d _K1, _K2;
    Vec4d _D1, _D2;
    if (!K1.empty()) K1.getMat().convertTo(_K1, CV_64FC1);
    if (!D1.empty()) D1.getMat().convertTo(_D1, CV_64FC1);
    if (!K2.empty()) K2.getMat().convertTo(_K2, CV_64FC1);
    if (!D2.empty()) D2.getMat().convertTo(_D2, CV_64FC1);

    // Load extrinsic parameters of matched chessboards
    std::vector<Vec3d> rvecs1(n_images), tvecs1(n_images), rvecs2(n_images), tvecs2(n_images);
    for(int i = 0; i < n_images; ++i){
        rvecs1[i] = Rckk1.getMat().at<Vec3d>(i);
        tvecs1[i] = Tckk1.getMat().at<Vec3d>(i);
        rvecs2[i] = Rckk2.getMat().at<Vec3d>(i);
        tvecs2[i] = Tckk2.getMat().at<Vec3d>(i);
    }


    intrinsicLeft.Init(Vec2d(_K1(0,0), _K1(1, 1)), Vec2d(_K1(0,2), _K1(1, 2)),
                       Vec4d(_D1[0], _D1[1], _D1[2], _D1[3]), _K1(0, 1) / _K1(0, 0));

    intrinsicRight.Init(Vec2d(_K2(0,0), _K2(1, 1)), Vec2d(_K2(0,2), _K2(1, 2)),
                        Vec4d(_D2[0], _D2[1], _D2[2], _D2[3]), _K2(0, 1) / _K2(0, 0));

    if ((flags & CALIB_FIX_INTRINSIC))
    {
        internal::CalibrateExtrinsics(objectPoints,  imagePoints1, intrinsicLeft, check_cond, thresh_cond, rvecs1, tvecs1);
        internal::CalibrateExtrinsics(objectPoints,  imagePoints2, intrinsicRight, check_cond, thresh_cond, rvecs2, tvecs2);
    }

    intrinsicLeft.isEstimate[0] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[1] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[2] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[3] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[4] = flags & (CALIB_FIX_SKEW | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[5] = flags & (CALIB_FIX_K1 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[6] = flags & (CALIB_FIX_K2 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[7] = flags & (CALIB_FIX_K3 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[8] = flags & (CALIB_FIX_K4 | CALIB_FIX_INTRINSIC) ? 0 : 1;

    intrinsicRight.isEstimate[0] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[1] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[2] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[3] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[4] = flags & (CALIB_FIX_SKEW | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[5] = flags & (CALIB_FIX_K1 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[6] = flags & (CALIB_FIX_K2 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[7] = flags & (CALIB_FIX_K3 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[8] = flags & (CALIB_FIX_K4 | CALIB_FIX_INTRINSIC) ? 0 : 1;

    intrinsicLeft_errors.isEstimate = intrinsicLeft.isEstimate;
    intrinsicRight_errors.isEstimate = intrinsicRight.isEstimate;

    std::vector<int> selectedParams;
    std::vector<int> tmp(6 * (n_images + 1), 1);
    selectedParams.insert(selectedParams.end(), intrinsicLeft.isEstimate.begin(), intrinsicLeft.isEstimate.end());
    selectedParams.insert(selectedParams.end(), intrinsicRight.isEstimate.begin(), intrinsicRight.isEstimate.end());
    selectedParams.insert(selectedParams.end(), tmp.begin(), tmp.end());

    //Init values for rotation and translation between two views
    cv::Mat om_list(1, n_images, CV_64FC3), T_list(1, n_images, CV_64FC3);
    cv::Mat om_ref, R_ref, T_ref, R1, R2;
    for (int image_idx = 0; image_idx < n_images; ++image_idx)
    {
        cv::Rodrigues(rvecs1[image_idx], R1);
        cv::Rodrigues(rvecs2[image_idx], R2);
        R_ref = R2 * R1.t();
        T_ref = cv::Mat(tvecs2[image_idx]) - R_ref * cv::Mat(tvecs1[image_idx]);
        cv::Rodrigues(R_ref, om_ref);
        om_ref.reshape(3, 1).copyTo(om_list.col(image_idx));
        T_ref.reshape(3, 1).copyTo(T_list.col(image_idx));
    }
    // Iron: can check the median value - does it close to our polycamera set?
    cv::Vec3d omcur = internal::median3d(om_list);
    cv::Vec3d Tcur  = internal::median3d(T_list);



    // 4 * n_points: ( left_x, left_y, right_x, right_y )
    cv::Mat J = cv::Mat::zeros(4 * n_points * n_images, 18 + 6 * (n_images + 1), CV_64FC1),
            e = cv::Mat::zeros(4 * n_points * n_images, 1, CV_64FC1), Jkk, ekk;
    cv::Mat J2_inv;

    for(int iter = 0; ; ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;

        J.create(4 * n_points * n_images, 18 + 6 * (n_images + 1), CV_64FC1);
        e.create(4 * n_points * n_images, 1, CV_64FC1);
        Jkk.create(4 * n_points, 18 + 6 * (n_images + 1), CV_64FC1);
        ekk.create(4 * n_points, 1, CV_64FC1);

        cv::Mat omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT;
        for (int image_idx = 0; image_idx < n_images; ++image_idx)
        {
            Jkk = cv::Mat::zeros(4 * n_points, 18 + 6 * (n_images + 1), CV_64FC1);

            cv::Mat object  = objectPoints.getMat(image_idx).clone();
            cv::Mat imageLeft  = imagePoints1.getMat(image_idx).clone();
            cv::Mat imageRight  = imagePoints2.getMat(image_idx).clone();
            cv::Mat jacobians, projected;

            //left camera jacobian
            // rvec1, tvec1: chessboard rotation and translation of left limage
            cv::Mat rvec = cv::Mat(rvecs1[image_idx]);
            cv::Mat tvec  = cv::Mat(tvecs1[image_idx]);
            internal::projectPoints(object, projected, rvec, tvec, intrinsicLeft, jacobians);



            cv::Mat(cv::Mat((imageLeft - projected).t()).reshape(1, 1).t()).copyTo(ekk.rowRange(0, 2 * n_points));
            jacobians.colRange(8, 11).copyTo(Jkk.colRange(24 + image_idx * 6, 27 + image_idx * 6).rowRange(0, 2 * n_points));
            jacobians.colRange(11, 14).copyTo(Jkk.colRange(27 + image_idx * 6, 30 + image_idx * 6).rowRange(0, 2 * n_points));
            jacobians.colRange(0, 2).copyTo(Jkk.colRange(0, 2).rowRange(0, 2 * n_points));
            jacobians.colRange(2, 4).copyTo(Jkk.colRange(2, 4).rowRange(0, 2 * n_points));
            jacobians.colRange(4, 8).copyTo(Jkk.colRange(5, 9).rowRange(0, 2 * n_points));
            jacobians.col(14).copyTo(Jkk.col(4).rowRange(0, 2 * n_points));

            //right camera jacobian
            internal::compose_motion(rvec, tvec, omcur, Tcur, omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT);
            rvec = cv::Mat(rvecs2[image_idx]);
            tvec  = cv::Mat(tvecs2[image_idx]);
            // compute jacobians of om/T (camera rotation and translation) and omckk/Tckk (chessboard rotation, translation)
            internal::projectPoints(object, projected, omr, Tr, intrinsicRight, jacobians);

            cv::Mat(cv::Mat((imageRight - projected).t()).reshape(1, 1).t()).copyTo(ekk.rowRange(2 * n_points, 4 * n_points));
            cv::Mat dxrdom = jacobians.colRange(8, 11) * domrdom + jacobians.colRange(11, 14) * dTrdom;
            cv::Mat dxrdT = jacobians.colRange(8, 11) * domrdT + jacobians.colRange(11, 14)* dTrdT;
            cv::Mat dxrdomckk = jacobians.colRange(8, 11) * domrdomckk + jacobians.colRange(11, 14) * dTrdomckk;
            cv::Mat dxrdTckk = jacobians.colRange(8, 11) * domrdTckk + jacobians.colRange(11, 14) * dTrdTckk;

            dxrdom.copyTo(Jkk.colRange(18, 21).rowRange(2 * n_points, 4 * n_points));
            dxrdT.copyTo(Jkk.colRange(21, 24).rowRange(2 * n_points, 4 * n_points));
            dxrdomckk.copyTo(Jkk.colRange(24 + image_idx * 6, 27 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
            dxrdTckk.copyTo(Jkk.colRange(27 + image_idx * 6, 30 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(0, 2).copyTo(Jkk.colRange(9 + 0, 9 + 2).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(2, 4).copyTo(Jkk.colRange(9 + 2, 9 + 4).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(4, 8).copyTo(Jkk.colRange(9 + 5, 9 + 9).rowRange(2 * n_points, 4 * n_points));
            jacobians.col(14).copyTo(Jkk.col(9 + 4).rowRange(2 * n_points, 4 * n_points));

            //check goodness of stereo pair
            double abs_max  = 0;
            for (int i = 0; i < 4 * n_points; i++)
            {
                if (fabs(ekk.at<double>(i)) > abs_max)
                {
                    abs_max = fabs(ekk.at<double>(i));
                }
            }
            if(abs_max > threshold)
            {
                std::cout << "abs_max:"<< abs_max <<", image_idx:" << image_idx << std::endl;

            }
            CV_Assert(abs_max < threshold); // bad stereo pair

            Jkk.copyTo(J.rowRange(image_idx * 4 * n_points, (image_idx + 1) * 4 * n_points));
            ekk.copyTo(e.rowRange(image_idx * 4 * n_points, (image_idx + 1) * 4 * n_points));
        }


        cv::Vec6d oldTom(Tcur[0], Tcur[1], Tcur[2], omcur[0], omcur[1], omcur[2]);

        //update all parameters
        internal::subMatrix(J, J, selectedParams, std::vector<int>(J.rows, 1));
        cv::Mat J2 = J.t() * J;
        J2_inv = J2.inv(); // AMSD_IRON_WIDE_FOV_FIX
        int a = cv::countNonZero(intrinsicLeft.isEstimate);
        int b = cv::countNonZero(intrinsicRight.isEstimate);
        cv::Mat deltas = J2_inv * J.t() * e;
        intrinsicLeft = intrinsicLeft + deltas.rowRange(0, a);
        intrinsicRight = intrinsicRight + deltas.rowRange(a, a + b);
        omcur = omcur + cv::Vec3d(deltas.rowRange(a + b, a + b + 3));
        Tcur = Tcur + cv::Vec3d(deltas.rowRange(a + b + 3, a + b + 6));
        for (int image_idx = 0; image_idx < n_images; ++image_idx)
        {
            rvecs1[image_idx] = cv::Mat(cv::Mat(rvecs1[image_idx]) + deltas.rowRange(a + b + 6 + image_idx * 6, a + b + 9 + image_idx * 6));
            tvecs1[image_idx] = cv::Mat(cv::Mat(tvecs1[image_idx]) + deltas.rowRange(a + b + 9 + image_idx * 6, a + b + 12 + image_idx * 6));
        }

        cv::Vec6d newTom(Tcur[0], Tcur[1], Tcur[2], omcur[0], omcur[1], omcur[2]);
        change = cv::norm(newTom - oldTom) / cv::norm(newTom);
    }

    double rms = 0;
    const Vec2d* ptr_e = e.ptr<Vec2d>();
    for (size_t i = 0; i < e.total() / 2; i++)
    {
        rms += ptr_e[i][0] * ptr_e[i][0] + ptr_e[i][1] * ptr_e[i][1];
    }

    rms /= ((double)e.total() / 2.0);
    rms = sqrt(rms);

    _K1 = Matx33d(intrinsicLeft.f[0], intrinsicLeft.alpha * intrinsicLeft.f[0], intrinsicLeft.c[0],
                                  0 ,                       intrinsicLeft.f[1], intrinsicLeft.c[1],
                                  0 ,                                       0 ,                 1);

    _K2 = Matx33d(intrinsicRight.f[0], intrinsicRight.alpha * intrinsicRight.f[0], intrinsicRight.c[0],
                                   0 ,                        intrinsicRight.f[1], intrinsicRight.c[1],
                                   0 ,                                         0 ,                  1);

    Mat _R;
    Rodrigues(omcur, _R);

    if (K1.needed()) cv::Mat(_K1).convertTo(K1, K1.empty() ? CV_64FC1 : K1.type());
    if (K2.needed()) cv::Mat(_K2).convertTo(K2, K2.empty() ? CV_64FC1 : K2.type());
    if (D1.needed()) cv::Mat(intrinsicLeft.k).convertTo(D1, D1.empty() ? CV_64FC1 : D1.type());
    if (D2.needed()) cv::Mat(intrinsicRight.k).convertTo(D2, D2.empty() ? CV_64FC1 : D2.type());
    if (R.needed()) _R.convertTo(R, R.empty() ? CV_64FC1 : R.type());
    if (T.needed()) cv::Mat(Tcur).convertTo(T, T.empty() ? CV_64FC1 : T.type());

    // update chessboard rotation and translation
    for (int image_idx = 0; image_idx < n_images; ++image_idx)
    {
        cv::Mat omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT;
        cv::Mat rvec = cv::Mat(rvecs1[image_idx]);
        cv::Mat tvec  = cv::Mat(tvecs1[image_idx]);
        internal::compose_motion(rvec, tvec, omcur, Tcur, omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT);
        Rckk1.getMat().at<Vec3d>(image_idx) = cv::Mat(rvecs1[image_idx]);
        Tckk1.getMat().at<Vec3d>(image_idx) = cv::Mat(tvecs1[image_idx]);
        Rckk2.getMat().at<Vec3d>(image_idx) = omr;
        Tckk2.getMat().at<Vec3d>(image_idx) = Tr;
    }

    return rms;
}


void sphericalRectifyE(Mat& E, Mat& R, Mat& R1, Mat& R2){
    Mat w, u, vt;
    SVD::compute(E, w, u, vt);
    Mat epi21 = u;
    Mat epi22 = -u;
    Mat epi11 = vt.t();
    Mat epi12 = -vt.t();
    epi11 = epi11.colRange(2, 3);
    epi12 = -epi11;
    epi21 = epi21.colRange(2, 3);
    epi22 = epi22.colRange(2, 3);
    epi11 = epi11 / norm(epi11);
    epi12 = epi12 / norm(epi12);
    epi21 = epi21 / norm(epi21);
    epi22 = epi22 / norm(epi22);
#if 1
    double max_val = abs(epi11.at<double>(0));
    int ind = 0;
    for(int i = 1; i < 3; ++i){
        ind = (max_val < abs(epi11.at<double>(i))) ? i : ind;
    }
    if (epi11.at<double>(ind) >= 0){
        Mat epi = epi11;
        epi11 = epi12;
        epi12 = epi;
    }
#endif
    Mat up = Mat::zeros(3, 1, CV_64FC1);
#if 0
    up.at<double>(1) = 1.0;
    Mat r3 = epi11;
    Mat epi11_t = epi11.t();
    Mat scale = epi11_t * up;
    Mat r2 = up - scale.at<double>(0) * epi11;
    Mat r1 = r2.cross(r3);
#endif
#if 1
    // opencv coordinate
    up.at<double>(2) = -1.0;
    Mat r2 = -epi11;
    Mat epi11_t = epi11.t();
    Mat scale = epi11_t * up;
    Mat r3 = up - scale.at<double>(0) * epi11;
    Mat r1 = r2.cross(r3);
#endif
    R1 = Mat::zeros(3, 3, CV_64FC1);
    r1 /= norm(r1);
    r2 /= norm(r2);
    r3 /= norm(r3);
    for(int r = 0; r < 3; ++r){
        R1.at<double>(r, 0) = r1.at<double>(r);
        R1.at<double>(r, 1) = r2.at<double>(r);
        R1.at<double>(r, 2) = r3.at<double>(r);
    }
    R2 = R * R1;
#if 0
    Mat product = R1.t() * epi11;
    cout << "up:"<<up.at<double>(0)<<","<<up.at<double>(1)<<","<<up.at<double>(2)<<endl;
    cout << "epipole:"<<epi11.at<double>(0)<<","<<epi11.at<double>(1)<<","<<epi11.at<double>(2)<<endl;
    cout << "R1 * epi11:"<<product.at<double>(0)<<","<<product.at<double>(1)<<","<<product.at<double>(2)<<endl;
    product = R2.t() * epi21;
    cout << "R2 * epi21:"<<product.at<double>(0)<<","<<product.at<double>(1)<<","<<product.at<double>(2)<<endl;
    cout<<"epi11:"<<epi11.at<double>(0)<<","<<epi11.at<double>(1)<<","<<epi11.at<double>(2)<<endl;
    cout<<"epi21:"<<epi21.at<double>(0)<<","<<epi21.at<double>(1)<<","<<epi21.at<double>(2)<<endl;
#endif

}

void sphericalRectifyRT(Mat& R, Mat& T, Mat& R1, Mat& R2){
    Mat epi11 = -R.t()*T;
    epi11 /= norm(epi11);
    Mat up = Mat::zeros(3, 1, CV_64FC1);
#if 0
    up.at<double>(1) = 1.0;
    Mat r3 = epi11;
    Mat epi11_t = epi11.t();
    Mat scale = epi11_t * up;
    Mat r2 = up - scale.at<double>(0) * epi11;
    Mat r1 = r2.cross(r3);
#endif
#if 1
    // opencv coordinate
    up.at<double>(2) = -1.0;
    Mat r2 = -epi11;
    Mat epi11_t = epi11.t();
    Mat scale = epi11_t * up;
    Mat r3 = up - scale.at<double>(0) * epi11;
    Mat r1 = r2.cross(r3);
#endif
    R1 = Mat::zeros(3, 3, CV_64FC1);
    r1 /= norm(r1);
    r2 /= norm(r2);
    r3 /= norm(r3);
    for(int r = 0; r < 3; ++r){
        R1.at<double>(r, 0) = r1.at<double>(r);
        R1.at<double>(r, 1) = r2.at<double>(r);
        R1.at<double>(r, 2) = r3.at<double>(r);
    }
    R2 = R * R1;
#if 0
    Mat product = R1.t() * epi11;
    cout << "up:"<<up.at<double>(0)<<","<<up.at<double>(1)<<","<<up.at<double>(2)<<endl;
    cout << "epipole:"<<epi11.at<double>(0)<<","<<epi11.at<double>(1)<<","<<epi11.at<double>(2)<<endl;
    cout << "R1 * epi11:"<<product.at<double>(0)<<","<<product.at<double>(1)<<","<<product.at<double>(2)<<endl;
    product = R2.t() * epi21;
    cout << "R2 * epi21:"<<product.at<double>(0)<<","<<product.at<double>(1)<<","<<product.at<double>(2)<<endl;
    cout<<"epi11:"<<epi11.at<double>(0)<<","<<epi11.at<double>(1)<<","<<epi11.at<double>(2)<<endl;
    cout<<"epi21:"<<epi21.at<double>(0)<<","<<epi21.at<double>(1)<<","<<epi21.at<double>(2)<<endl;
#endif

}

void sphericalRectifyRT(Mat& R, Mat& T, Mat& R1, Mat& R2, Vec2d& theta_range, Vec2d& phi_range){
    Mat epi11 = -R.t()*T;
    epi11 /= norm(epi11);
    Mat up = Mat::zeros(3, 1, CV_64FC1);
#if 0
    up.at<double>(1) = 1.0;
    Mat r3 = epi11;
    Mat epi11_t = epi11.t();
    Mat scale = epi11_t * up;
    Mat r2 = up - scale.at<double>(0) * epi11;
    Mat r1 = r2.cross(r3);
#endif
#if 1
    // opencv coordinate
    up.at<double>(2) = -1.0;
    Mat r2 = -epi11;
    Mat epi11_t = epi11.t();
    Mat scale = epi11_t * up;
    Mat r3 = up - scale.at<double>(0) * epi11;
    Mat r1 = r2.cross(r3);
#endif
    R1 = Mat::zeros(3, 3, CV_64FC1);
    r1 /= norm(r1);
    r2 /= norm(r2);
    r3 /= norm(r3);
    for(int r = 0; r < 3; ++r){
        R1.at<double>(r, 0) = r1.at<double>(r);
        R1.at<double>(r, 1) = r2.at<double>(r);
        R1.at<double>(r, 2) = r3.at<double>(r);
    }
    R2 = R * R1;
    //determine range of co-latitude and longitude in rotated spherical coordinate system
    Mat axis = Mat::zeros(3, 1, CV_64FC1);
    axis.at<double>(2) = 1.0;
    Mat ray1 = R1.t() * axis;
    // project axis of second sphere into first sphere and rotate
    Mat ray2 = R1.t() * R.t() * axis;
    ray1 /= norm(ray1);
    ray2 /= norm(ray2);
    double xs = ray1.at<double>(0);
    double ys = ray1.at<double>(1);
    double zs = ray1.at<double>(2);
    double theta = acos(zs);
    double phi[4];
    double val = atan(ys/xs);
    if(xs < 0.0)
        val += CV_PI;
    else if(ys < 0.0)
        val += CV_PI * 2;
    phi[0] = val - CV_PI/2;
    phi[1] = val + CV_PI/2;
    theta_range[0] = theta;
    xs = ray2.at<double>(0);
    ys = ray2.at<double>(1);
    zs = ray2.at<double>(2);
    theta = acos(zs);
    val = atan(ys/xs);
    if(xs < 0.0)
        val += CV_PI;
    else if(ys < 0.0)
        val += CV_PI * 2;
    phi[2] = val - CV_PI/2;
    phi[3] = val + CV_PI/2;
    double emp = 0.01;
    theta_range[1] = (theta_range[0] < theta) ? theta : theta_range[0];
    theta_range[0] = (theta_range[0] >= theta) ? theta : theta_range[0];
    theta_range[0] -= emp;
    theta_range[1] += emp;
    // compute
    phi_range[0] = (phi[0] < phi[2]) ? phi[2] : phi[0];
    phi_range[1] = (phi[1] < phi[3]) ? phi[1] : phi[3];
}

void genPixelRay(Mat& imagePoints, Mat& rays, Mat& K, Mat& D)
{
    Vec2d f, c;
    double alpha;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K;
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
        alpha = camMat(0, 1)/camMat(0, 0);
    }
    else
    {
        Matx33d camMat = K;
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
        alpha = camMat(0, 1)/camMat(0, 0);
    }

    Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.ptr<Vec4f>(): *D.ptr<Vec4d>();

    // start undistorting
    const cv::Vec2f* srcf = imagePoints.ptr<cv::Vec2f>();
    const cv::Vec2d* srcd = imagePoints.ptr<cv::Vec2d>();
    cv::Vec3f* dstf = rays.ptr<cv::Vec3f>();
    cv::Vec3d* dstd = rays.ptr<cv::Vec3d>();

    size_t n = rays.total();
    int sdepth = imagePoints.depth();
//#pragma omp parallel for
    for(size_t i = 0; i < n; i++ )
    {
        Vec2d pi = sdepth == CV_32F ? (Vec2d)srcf[i] : srcd[i];  // image point
        Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point
        // add by iron: consider skew factor
        pw[0] -= alpha * pw[1];
        double scale = 1.0;
        // r = f * theta
        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);
        if (theta_d > 1e-8)
        {
            // compensate distortion iteratively
            double theta = theta_d;
            for(int j = 0; j < 10; j++ )
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                theta = theta_d / (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8);
            }

            scale = std::tan(theta) / theta_d;

            //cout << "theta:"<<theta<<", thetad:"<<theta_d<<endl;
        }

 #if 0
        Vec2d pu = pw * scale; //undistorted point
        Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final
#endif
#if 1
        // added by Iron: to hangle over 180 degree projection
        Vec2d pu;
        Vec3d pr;
        if(scale >= 0.0){
            pu = pw * scale;
            pr = Vec3d(pu[0], pu[1], 1.0);
        }
        else{
            pu = pw * scale;
            pu[0] = -pu[0];
            pu[1] = -pu[1];
            pr = Vec3d(pu[0], pu[1], -1.0);
        }


#endif
        // normalize the ray
        double length = norm(pr);
        pr = pr/length;

        if( sdepth == CV_32F )
            dstf[i] = pr;
        else
            dstd[i] = pr;
    }
}

void overlapMask(Mat& mask_left, Mat& mask_right, Size mask_size,
                 Mat& K1, Mat& K2, Mat& D1, Mat& D2, Mat& R, double fov){
    // compute overlap mask according to relative rotation
    mask_left = Mat::zeros(mask_size, CV_8UC1);
    mask_right = Mat::zeros(mask_size, CV_8UC1);
    int rows = mask_size.height;
    int cols = mask_size.width;
    // compute rays of each pixel
    Mat points = Mat::zeros(1, rows * cols, CV_64FC2);
    Mat ray_left = Mat::zeros(1, rows * cols, CV_64FC3);
    Mat ray_right = Mat::zeros(1, rows * cols, CV_64FC3);
    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            points.at<Vec2d>(r * cols + c) = Vec2d(c, r);
        }
    }
    genPixelRay(points, ray_left, K1, D1);
    genPixelRay(points, ray_right, K2, D2);
    // check each ray, if the angle between ray and z axis of the another frame is larger than the angle between two frames,
    // the pixel is discarded
    // compute the angle between two frames
    Vec3d om;
    Rodrigues(R, om);
    double angle = norm(om);
    double emp = 0.01;
    // check left image
    // rotate principal axis of left image to right image
    Mat axe = Mat::zeros(3, 1, CV_64FC1);
    axe.at<double>(2) = 1.0;
    Mat axe_r = R.t() * axe;
    // compute radius of fisheye circle
    Mat objectPoint = Mat::zeros(1, 1, CV_64FC3);
    Mat imagePoint = Mat::zeros(1, 1, CV_64FC2);
    Vec2d cp_left = Vec2d(K1.at<double>(0,2), K1.at<double>(1,2));
    Vec2d cp_right = Vec2d(K2.at<double>(0,2), K2.at<double>(1,2));
    double theta = fov / 360.0 * CV_PI;
    objectPoint.at<Vec3d>(0)[0] = sin(theta);
    objectPoint.at<Vec3d>(0)[2] = cos(theta);
    cv::widefov::projectPoint(objectPoint, imagePoint, K1, D1, K1.at<double>(0, 1)/K1.at<double>(0, 0));
    Vec2d pt = imagePoint.at<Vec2d>(0);

    double radius_left = norm(pt-cp_left);
    cv::widefov::projectPoint(objectPoint, imagePoint, K2, D2, K2.at<double>(0, 1)/K2.at<double>(0, 0));
    pt = imagePoint.at<Vec2d>(0);

    double radius_right = norm(pt-cp_right);
    double ita = RADIUS_OFFSET;
//#pragma omp parallel for
    for(int r = 0; r < mask_left.rows; ++r){
        for(int c = 0; c < mask_left.cols; ++c){
            pt = Vec2d(c, r);
            if(norm(pt-cp_left) < radius_left + ita){
                // compute angle between ray and the rotated principal axe
                double product = 0;
                for(int k = 0; k < 3; ++k)
                    product += ray_left.at<Vec3d>(r*cols+c)[k] * axe_r.at<double>(k);
                if(acos(product) < theta){
                    mask_left.at<uchar>(r, c) = 255;
                }
            }
        }
    }
    // check left image
    // rotate principal axis of left image to right image
    axe_r = R * axe;
//#pragma omp parallel for
    for(int r = 0; r < mask_right.rows; ++r){
        for(int c = 0; c < mask_right.cols; ++c){
            // compute angle between ray and the rotated principal axe
            pt = Vec2d(c, r);
            if(norm(pt-cp_right) < radius_right + ita){
                double product = 0;
                for(int k = 0; k < 3; ++k)
                    product += ray_right.at<Vec3d>(r*cols+c)[k] * axe_r.at<double>(k);
                if(acos(product) < theta)
                    mask_right.at<uchar>(r, c) = 255;
            }
        }
    }
}

void stereoRectify(string image_file_left, string image_file_right,
                   string rect_file_left, string rect_file_right,
                   Mat& R, Mat& T, Mat& K1, Mat& K2, Mat& D1, Mat& D2,
                   Mat& mask_left_equi, Mat& mask_right_equi, Rect& rect_bb_1, Rect& rect_bb_2,
                   double fov){

    Mat img_left = imread(image_file_left);
    Mat img_right = imread(image_file_right);
    char filename[256];
    Mat mask_left, mask_right;
    overlapMask(mask_left, mask_right, img_left.size(), K1, K2, D1, D2, R, fov);
    sprintf(filename, "fish-mask-%s.jpg", rect_file_left.c_str());
    imwrite(filename, mask_left);
    sprintf(filename, "fish-mask-%s.jpg", rect_file_right.c_str());
    imwrite(filename, mask_right);
#if 0
    for(size_t r = 0; r < img_left.rows; ++r){
        for(size_t c = 0; c < img_left.cols; ++c){
            img_left.at<Vec3b>(r, c) = (mask_left.at<uchar>(r, c) == 0) ? Vec3b(0, 0, 0) : img_left.at<Vec3b>(r, c);
            img_right.at<Vec3b>(r, c) = (mask_right.at<uchar>(r, c) == 0) ? Vec3b(0, 0, 0) : img_right.at<Vec3b>(r, c);
        }
    }
#endif

#if 0
    cout << T.at<double>(0)<<","<<T.at<double>(1)<<","<<T.at<double>(2)<<endl;
    Vec3d om;
    Rodrigues(R, om);
    double angle = norm(om);
    cout << om[0]/angle<<","<<om[1]/angle<<","<<om[2]/angle<<". angle:"<<angle * 180/3.141526<<endl;
#endif
#if 1
    // spherical rectification implementation: iron
    Mat img_left_equi, img_right_equi;
    // draw aligned images
    Mat E;
    computeEssentialMatrix(R, T, E);
    Mat R1, R2;
    Vec2d theta_interval, phi_interval;
    sphericalRectifyRT(R, T, R1, R2, theta_interval, phi_interval);
    //Mat mask_left_equi, mask_right_equi;
    Mat rect_mat_1, rect_mat_2;
    equirect2Fisheye(R1, K1, D1, rect_mat_1, fov, mask_left, mask_left_equi);
    equirect2Fisheye(R2, K2, D2, rect_mat_2, fov, mask_right, mask_right_equi);
    rectBoundingBox(rect_mat_1, rect_bb_1);
    rectBoundingBox(rect_mat_2, rect_bb_2);
    // unify bounding box in same height
    int r1 = rect_bb_1.y;
    int r2 = rect_bb_1.y + rect_bb_1.height;
    int r3 = rect_bb_2.y;
    int r4 = rect_bb_2.y + rect_bb_2.height;
    rect_bb_1.y = rect_bb_2.y = (r1 < r3) ? r3 : r1;
    rect_bb_1.height = rect_bb_2.height = (r2 < r4) ? r2 - rect_bb_1.y : r4 - rect_bb_1.y;

    // crop mask_equi by bounding box
    mask_left_equi = mask_left_equi.rowRange(rect_bb_1.y, rect_bb_1.y+rect_bb_1.height+1).colRange(rect_bb_1.x, rect_bb_1.x+rect_bb_1.width+1);
    mask_right_equi = mask_right_equi.rowRange(rect_bb_2.y, rect_bb_2.y+rect_bb_2.height+1).colRange(rect_bb_2.x, rect_bb_2.x+rect_bb_2.width+1);
    // generate rectified image

    remapImage(img_left, img_left_equi, rect_mat_1);
    remapImage(img_right, img_right_equi, rect_mat_2);

#if 0
    // memory issue: crash here if images are large
    Mat R1_inv = R1.t();
    Mat R2_inv = R2.t();
    cout << "fisheye 2 equirectangular images"<<endl;
    fisheye2Equirect(R1_inv, K1, D1, fish_mat_1, mask_left);
    fisheye2Equirect(R2_inv, K2, D2, fish_mat_2, mask_right);
    Mat img_left_fish, img_right_fish;
    remapImage(img_left_equi, img_left_fish, fish_mat_1);
    remapImage(img_right_equi, img_right_fish, fish_mat_2);
    //sphereImage(img_left, img_left_equi, R1, K1, D1);
    //sphereImage(img_right, img_right_equi, R2, K2, D2);
    for(int r = 0; r < img_left.rows; ++r){
        for(int c = 0; c < img_right.cols; ++c){
            img_left.at<Vec3b>(r, c) = (mask_left.at<uchar>(r, c) == 0) ? Vec3b(0, 0, 0): img_left.at<Vec3b>(r, c);
            img_right.at<Vec3b>(r, c) = (mask_right.at<uchar>(r, c) == 0) ? Vec3b(0, 0, 0): img_right.at<Vec3b>(r, c);
        }
    }


    // draw bounding box on remapped images
    r1 = rect_bb_1.y;
    r2 = rect_bb_1.y + rect_bb_1.height;
    int c1 = rect_bb_1.x;
    int c2 = rect_bb_1.x + rect_bb_1.width;
    line(img_left_equi, Point(c1, r1), Point(c2, r1), Scalar(0, 255, 0), 4);
    line(img_left_equi, Point(c1, r1), Point(c1, r2), Scalar(0, 255, 0), 4);
    line(img_left_equi, Point(c2, r2), Point(c2, r1), Scalar(0, 255, 0), 4);
    line(img_left_equi, Point(c2, r2), Point(c1, r2), Scalar(0, 255, 0), 4);

    r1 = rect_bb_2.y;
    r2 = rect_bb_2.y + rect_bb_2.height;
    c1 = rect_bb_2.x;
    c2 = rect_bb_2.x + rect_bb_2.width;
    line(img_right_equi, Point(c1, r1), Point(c2, r1), Scalar(0, 255, 0), 4);
    line(img_right_equi, Point(c1, r1), Point(c1, r2), Scalar(0, 255, 0), 4);
    line(img_right_equi, Point(c2, r2), Point(c2, r1), Scalar(0, 255, 0), 4);
    line(img_right_equi, Point(c2, r2), Point(c1, r2), Scalar(0, 255, 0), 4);
    sprintf(filename, "fish-%s.jpg", rect_file_left.c_str());
    imwrite(filename, img_left_fish);
    sprintf(filename, "fish-%s.jpg", rect_file_right.c_str());
    imwrite(filename, img_right_fish);
#endif
    sprintf(filename, "equi-rect-%s.jpg", rect_file_left.c_str());
    imwrite(filename, img_left_equi);
    sprintf(filename, "equi-rect-%s.jpg", rect_file_right.c_str());
    imwrite(filename, img_right_equi);

    sprintf(filename, "crop-img-%s.jpg", rect_file_left.c_str());
    imwrite(filename, img_left);
    sprintf(filename, "crop-img-%s.jpg", rect_file_right.c_str());
    imwrite(filename, img_right);
#endif
#if 0
    // image rectification (opencv not uses spherical rectification)
    Mat R1, R2, P1, P2, Q;
    Size undist_size = Size(img_left.cols, img_left.rows);
    stereoRectify(K1, D1, K2, D2, img_left.size(),
        R, T, R1, R2, P1, P2, Q, 0, undist_size, 0, 2.0);
    Mat img_left_undist, img_right_undist;
    Mat map1, map2;
    initUndistortRectifyMap(K1, D1, R1, P1, undist_size, CV_16SC2, map1, map2 );
    remap(img_left, img_left_undist, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
    initUndistortRectifyMap(K2, D2, R2, P2, undist_size, CV_16SC2, map1, map2 );
    remap(img_right, img_right_undist, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
    sprintf(filename, "rect-%s.jpg", rect_file_left.c_str());
    imwrite(filename, img_left_undist);
    sprintf(filename, "rect-%s.jpg", rect_file_right.c_str());
    imwrite(filename, img_right_undist);
#endif
}

void computeEssentialMatrix(Mat& R, Mat& T, Mat& E){
    // X2 = RX1+T
    // E = txR
    Mat tx = Mat::zeros(3, 3, CV_64FC1);
    tx.at<double>(0, 1) = -T.at<double>(2);
    tx.at<double>(0, 2) = T.at<double>(1);
    tx.at<double>(1, 0) = T.at<double>(2);
    tx.at<double>(1, 2) = -T.at<double>(0);
    tx.at<double>(2, 0) = -T.at<double>(1);
    tx.at<double>(2, 1) = T.at<double>(0);
    E = tx * R;
}

void equirect2Fisheye(Mat& R, Mat& K, Mat& D, Mat& maps, double fov, Mat& mask, Mat& mask_equi){
    int w = mask.rows;
    int h = mask.cols;
    maps = -Mat::ones(h, w, CV_64FC2);
    mask_equi = Mat::zeros(h, w, CV_8UC1);

#pragma omp parallel for
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c){
            // r : azimuth sampling, c: co-latitude sampling
            // (r, c) --> (azimuth (phi),  co-latitude (theta))
            double theta = (double)c / (double) w * CV_PI; // (0-PI/2)
            double phi = (double)r / (double) h * 2 * CV_PI; // (0- 2 * PI)
            // (longitude, latitude) --> spherical coordinates
#if 0
            double xs = cos(phi) * sin(theta);
            double ys = sin(phi) * sin(theta);
            double zs = cos(theta);
#endif
#if 1
            // opencv coordinates
            double zs = sin(theta) * cos(phi);
            double ys = -cos(theta);
            double xs = sin(theta) * sin(phi);
#endif
            // spherical coordinates --> rotate points
            Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
            ps_vec.at<double>(0) = xs;
            ps_vec.at<double>(1) = ys;
            ps_vec.at<double>(2) = zs;

            // allow maximum projection angle : 100 degree
            Mat pr_vec = R * ps_vec;
            if(acos(pr_vec.at<double>(2)) > fov/360.0 * CV_PI)
                continue;
            Mat pr = Mat::zeros(1, 1, CV_64FC3);
            pr.at<Vec3d>(0)[0] = pr_vec.at<double>(0);
            pr.at<Vec3d>(0)[1] = pr_vec.at<double>(1);
            pr.at<Vec3d>(0)[2] = pr_vec.at<double>(2);
            Mat pf = Mat::zeros(1, 1, CV_64FC2);
            cv::widefov::projectPoint(pr, pf, K, D, K.at<double>(0,1) / K.at<double>(0, 0));
            int xf = cvRound(pf.at<Vec2d>(0)[0]);
            int yf = cvRound(pf.at<Vec2d>(0)[1]);
            if(xf < mask.cols && xf >= 0 && yf < mask.rows && yf >= 0){
                if(mask.at<uchar>(yf, xf) == 255)
                {
                    maps.at<Vec2d>(r, c) = pf.at<Vec2d>(0);
                    mask_equi.at<uchar>(r, c) = 255;
                }
            }
        }
    }
}

void equirect2Fisheye(Mat& R, Mat& K, Mat& D, Mat& maps, Mat& mask, Mat& mask_equi){

    int w = (double) mask.rows;
    int h = (double) mask.cols;
    maps = -Mat::ones(h, w, CV_64FC2);
    mask_equi = Mat::zeros(h, w, CV_8UC1);
#pragma omp parallel for
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c){
            // r : azimuth sampling, c: co-latitude sampling
            // (r, c) --> (azimuth (phi),  co-latitude (theta))
            double theta = (double)c / (double) w * CV_PI;
            double phi = (double)r / (double) h * 2 * CV_PI;

#if 0
            // (longitude, latitude) --> spherical coordinates
            double xs = cos(phi) * sin(theta);
            double ys = sin(phi) * sin(theta);
            double zs = cos(theta);
#endif
#if 1
            // opencv coordinates
            double zs = sin(theta) * cos(phi);
            double ys = -cos(theta);
            double xs = sin(theta) * sin(phi);
#endif

            // spherical coordinates --> fisheye image coordinates
            Mat ps = Mat::zeros(1, 1, CV_64FC3);
            ps.at<Vec3d>(0) = Vec3d(xs, ys, zs);
            // spherical coordinates --> rotate points
            Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
            ps_vec.at<double>(0) = xs;
            ps_vec.at<double>(1) = ys;
            ps_vec.at<double>(2) = zs;
            Mat pr_vec = R * ps_vec;
            if(acos(pr_vec.at<double>(2)) > 100.0/180.0 * CV_PI)
                continue;
            Mat pr = Mat::zeros(1, 1, CV_64FC3);
            pr.at<Vec3d>(0)[0] = pr_vec.at<double>(0);
            pr.at<Vec3d>(0)[1] = pr_vec.at<double>(1);
            pr.at<Vec3d>(0)[2] = pr_vec.at<double>(2);
            Mat pf = Mat::zeros(1, 1, CV_64FC2);
            cv::widefov::projectPoint(pr, pf, K, D, K.at<double>(0,1) / K.at<double>(0, 0));
            int xf = cvRound(pf.at<Vec2d>(0)[0]);
            int yf = cvRound(pf.at<Vec2d>(0)[1]);
            if(xf < mask.cols && xf >= 0 && yf < mask.rows && yf >= 0){
                if(mask.at<uchar>(yf, xf) == 255){
                    maps.at<Vec2d>(r, c) = pf.at<Vec2d>(0);
                    mask_equi.at<uchar>(r, c) = 255;
                }
            }
        }
    }
}

void equirect2FisheyeWithBB(Mat& R, Mat& K, Mat& D, Mat& maps,
                            Mat& mask_equi, Rect& rect_bb, Size img_size){
    //int w = rect_bb.width+rect_bb.x+1;
    //int h = rect_bb.height+rect_bb.y+1;
    int w = img_size.height;
    int h = img_size.width;
    maps = -Mat::ones(rect_bb.height+1, rect_bb.width+1, CV_64FC2);
    Mat R_inv = R.t();
    int r1 = rect_bb.y;
    int r2 = rect_bb.y + rect_bb.height;
    int c1 = rect_bb.x;
    int c2 = rect_bb.x + rect_bb.width;
#pragma omp parallel for
    for(int r = r1; r <= r2; ++r){
        for(int c = c1; c <= c2; ++c){
            //if(mask_equi.at<uchar>(r-r1, c-c1) < 128)
                //continue;
            // r : azimuth sampling, c: co-latitude sampling
            // (r, c) --> (azimuth (phi),  co-latitude (theta))
            double theta = (double)c / (double) w * CV_PI; // (0-PI)
            double phi = (double)r / (double) h * 2 * CV_PI; // (0- 2 * PI)
            // (longitude, latitude) --> spherical coordinates
#if 0
            double xs = cos(phi) * sin(theta);
            double ys = sin(phi) * sin(theta);
            double zs = cos(theta);
#endif
#if 1
            // opencv coordinates
            double zs = sin(theta) * cos(phi);
            double ys = -cos(theta);
            double xs = sin(theta) * sin(phi);
#endif

            // spherical coordinates --> fisheye image coordinates
            Mat ps = Mat::zeros(1, 1, CV_64FC3);
            ps.at<Vec3d>(0) = Vec3d(xs, ys, zs);
            // spherical coordinates --> rotate points
            Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
            ps_vec.at<double>(0) = xs;
            ps_vec.at<double>(1) = ys;
            ps_vec.at<double>(2) = zs;

            Mat pr_vec = R * ps_vec;
            Mat pr = Mat::zeros(1, 1, CV_64FC3);
            pr.at<Vec3d>(0)[0] = pr_vec.at<double>(0);
            pr.at<Vec3d>(0)[1] = pr_vec.at<double>(1);
            pr.at<Vec3d>(0)[2] = pr_vec.at<double>(2);
            Mat pf = Mat::zeros(1, 1, CV_64FC2);
            cv::widefov::projectPoint(pr, pf, K, D, K.at<double>(0,1) / K.at<double>(0, 0));
            int xf = cvRound(pf.at<Vec2d>(0)[0]);
            int yf = cvRound(pf.at<Vec2d>(0)[1]);
            if(xf < img_size.width && xf >= 0 && yf < img_size.height && yf >= 0){
                maps.at<Vec2d>(r-r1, c-c1) = pf.at<Vec2d>(0);
            }
        }
    }
}

#if 0
void equirect2FisheyeWithBB(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& maps,
                            cv::Mat& mask_equi, cv::Rect& rect_bb, cv::Size img_size,
                            cv::Vec2d& theta_interval, cv::Vec2d& phi_interval){
    double theta_start = theta_interval[0];
    double theta_range = theta_interval[1] - theta_interval[0];
    double phi_start = phi_interval[0];
    double phi_range = phi_interval[1] - phi_interval[0];
    int w = rect_bb.width+rect_bb.x+1;
    int h = rect_bb.height+rect_bb.y+1;
    maps = -Mat::ones(rect_bb.height+1, rect_bb.width+1, CV_64FC2);
    Mat R_inv = R.t();
    int r1 = rect_bb.y;
    int r2 = rect_bb.y + rect_bb.height;
    int c1 = rect_bb.x;
    int c2 = rect_bb.x + rect_bb.width;
#pragma omp parallel for
    for(int r = r1; r <= r2; ++r){
        for(int c = c1; c <= c2; ++c){
            if(mask_equi.at<uchar>(r-r1, c-c1) < 128)
                continue;
            // r : azimuth sampling, c: co-latitude sampling
            // (r, c) --> (azimuth (phi),  co-latitude (theta))
            double theta = (double)c / (double) w * CV_PI;
            double phi = (double)r / (double) h * 2 * CV_PI;
            // (longitude, latitude) --> spherical coordinates
            double xs = cos(phi) * sin(theta);
            double ys = sin(phi) * sin(theta);
            double zs = cos(theta);

            // spherical coordinates --> fisheye image coordinates
            Mat ps = Mat::zeros(1, 1, CV_64FC3);
            ps.at<Vec3d>(0) = Vec3d(xs, ys, zs);
            // spherical coordinates --> rotate points
            Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
            ps_vec.at<double>(0) = xs;
            ps_vec.at<double>(1) = ys;
            ps_vec.at<double>(2) = zs;

            Mat pr_vec = R * ps_vec;
            Mat pr = Mat::zeros(1, 1, CV_64FC3);
            pr.at<Vec3d>(0)[0] = pr_vec.at<double>(0);
            pr.at<Vec3d>(0)[1] = pr_vec.at<double>(1);
            pr.at<Vec3d>(0)[2] = pr_vec.at<double>(2);
            Mat pf = Mat::zeros(1, 1, CV_64FC2);
            cv::widefov::projectPoint(pr, pf, K, D, K.at<double>(0,1) / K.at<double>(0, 0));
            int xf = cvRound(pf.at<Vec2d>(0)[0]);
            int yf = cvRound(pf.at<Vec2d>(0)[1]);
            if(xf < img_size.width && xf >= 0 && yf < img_size.height && yf >= 0){
                maps.at<Vec2d>(r-r1, c-c1) = pf.at<Vec2d>(0);
            }
        }
    }
}
#endif

void fisheye2Equirect(Mat& R, Mat& K, Mat& D, Mat& maps, Mat& mask){
    maps = -Mat::ones(mask.size(), CV_64FC2);
    int h = mask.rows;
    int w = mask.cols;
    Size rect_size;
    rect_size.width = h;
    rect_size.height = w;
    // compute rays of each pixel
    Mat points = Mat::zeros(1, h*w, CV_64FC2);
    Mat rays = Mat::zeros(1, h*w, CV_64FC3);
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c)
            points.at<Vec2d>(r*w+c) = Vec2d(c, r);
    }
    genPixelRay(points, rays, K, D);
#pragma omp parallel for
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c){
            if(mask.at<uchar>(r, c) >= 128){
                Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
                ps_vec.at<double>(0) = rays.at<Vec3d>(r*w+c)[0];
                ps_vec.at<double>(1) = rays.at<Vec3d>(r*w+c)[1];
                ps_vec.at<double>(2) = rays.at<Vec3d>(r*w+c)[2];
                Mat pr_vec = R * ps_vec;
                double xs = pr_vec.at<double>(0);
                double ys = pr_vec.at<double>(1);
                double zs = pr_vec.at<double>(2);
#if 0
                double theta = acos(zs);
                double phi = atan(ys/xs);

                if(xs < 0)
                    phi += CV_PI;
                else if(ys < 0)
                    phi += 2 * CV_PI;
#endif
#if 1
                double theta = acos(-ys);
                double phi = atan(xs / zs);
                if(zs < 0.0)
                    phi += CV_PI;
                else if(xs < 0.0)
                    phi += 2 * CV_PI;
#endif
                Vec2d pf;
                pf[0] = theta * (double) rect_size.width / CV_PI;
                pf[1] = phi * (double) rect_size.height * 0.5 / CV_PI;
                int xf = cvRound(pf[0]);
                int yf = cvRound(pf[1]);
                if(xf < rect_size.width && xf >= 0 && yf < rect_size.height && yf >= 0)
                    maps.at<Vec2d>(r, c) = pf;
            }
        }
    }
}

void fisheye2EquirectWithBB(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& maps,
                            cv::Mat& mask, Rect& bb){
    maps = -Mat::ones(mask.size(), CV_64FC2);
    int h = mask.rows;
    int w = mask.cols;
    Size rect_size;
    rect_size.width = h;
    rect_size.height = w;
    // compute rays of each pixel
    Mat points = Mat::zeros(1, h*w, CV_64FC2);
    Mat rays = Mat::zeros(1, h*w, CV_64FC3);
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c)
            points.at<Vec2d>(r*w+c) = Vec2d(c, r);
    }
    genPixelRay(points, rays, K, D);
#pragma omp parallel for
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c){
            if(mask.at<uchar>(r, c) >= 128){
                Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
                ps_vec.at<double>(0) = rays.at<Vec3d>(r*w+c)[0];
                ps_vec.at<double>(1) = rays.at<Vec3d>(r*w+c)[1];
                ps_vec.at<double>(2) = rays.at<Vec3d>(r*w+c)[2];
                Mat pr_vec = R * ps_vec;
                double xs = pr_vec.at<double>(0);
                double ys = pr_vec.at<double>(1);
                double zs = pr_vec.at<double>(2);
#if 0
                double theta = acos(zs);
                double phi = atan(ys/xs);

                if(xs < 0)
                    phi += CV_PI;
                else if(ys < 0)
                    phi += 2 * CV_PI;
#endif
#if 1
                double theta = acos(-ys);
                double phi = atan(xs / zs);
                if(zs < 0.0)
                    phi += CV_PI;
                else if(xs < 0.0)
                    phi += 2 * CV_PI;
#endif
                Vec2d pf;
                pf[0] = theta * (double) rect_size.width / CV_PI;
                pf[1] = phi * (double) rect_size.height * 0.5 / CV_PI;
                pf[0] -= bb.x;
                pf[1] -= bb.y;
                int xf = cvRound(pf[0]);
                int yf = cvRound(pf[1]);
                if(xf < rect_size.width && xf >= 0 && yf < rect_size.height && yf >= 0)
                    maps.at<Vec2d>(r, c) = pf;
            }
        }
    }

}

void rectBoundingBox(Mat& rect_mat, Rect& rect_bb){
    rect_bb.x = rect_mat.cols;
    rect_bb.y = rect_mat.rows;
    rect_bb.width = 0;
    rect_bb.height = 0;
    for(int r = 0; r < rect_mat.rows; ++r){
        for(int c = 0; c < rect_mat.cols; ++c){
            if(rect_mat.at<Vec2d>(r, c)[0] >= 0.0){
                if(c < rect_bb.x){
                    if(rect_bb.x < rect_mat.cols)
                        rect_bb.width = rect_bb.x + rect_bb.width - c;
                    rect_bb.x = c;
                }
                else if(c > rect_bb.x + rect_bb.width)
                    rect_bb.width = c - rect_bb.x;
                if(r < rect_bb.y){
                    if(rect_bb.y < rect_mat.rows)
                        rect_bb.height = rect_bb.y + rect_bb.height - r;
                    rect_bb.y = r;
                }
                else if(r > rect_bb.y + rect_bb.height)
                    rect_bb.height = r - rect_bb.y;
            }
        }
    }
    cout << "bounding box:"<<rect_bb.x<<","<<rect_bb.y<<","<<rect_bb.width<<","<<rect_bb.height<<endl;
}

void remapImage(Mat& img, Mat& img_remap, Mat& remap_mat){
    img_remap = Mat::zeros(remap_mat.size(), CV_8UC3);
    for(int r = 0; r < remap_mat.rows; ++r){
        for(int c = 0; c < remap_mat.cols; ++c){
            if(remap_mat.at<Vec2d>(r, c)[0] >= 0.0){
                Vec2d xf = remap_mat.at<Vec2d>(r, c);
                // perform bilinear interpolation
#if 1
                // bilinear interpolation
                Vec2i xi[4];
                xi[0][0] = floor(xf[0]);
                xi[0][1] = floor(xf[1]);
                xi[3][0] = ceil(xf[0]);
                xi[3][1] = ceil(xf[1]);
                if(xi[0][0] >= 0 && xi[3][0] < img.cols && xi[0][1] >= 0 && xi[3][1] < img.rows){

                xi[1][0] = xi[3][0];
                xi[1][1] = xi[0][1];
                xi[2][0] = xi[0][0];
                xi[2][1] = xi[3][1];
                double alpha[4];
                alpha[0] = (1-(xf[0]-xi[0][0])) * (1-(xf[1]-xi[0][1]));
                alpha[1] = (xf[0]-xi[0][0]) * (1-(xf[1]-xi[0][1]));
                alpha[2] = (1-(xf[0]-xi[0][0])) * (xf[1]-xi[0][1]);
                alpha[3] = (xf[0]-xi[0][0]) * (xf[1]-xi[0][1]);
                Vec3d val(0.0, 0.0, 0.0);
                for(size_t i = 0; i < 4; ++i){
                    for(size_t k = 0; k < 3; ++k){
                        val[k] += alpha[i] * (double)img.at<Vec3b>(xi[i][1], xi[i][0])[k];
                    }
                }
                img_remap.at<Vec3b>(r, c)[0] = (uchar)val[0];
                img_remap.at<Vec3b>(r, c)[1] = (uchar)val[1];
                img_remap.at<Vec3b>(r, c)[2] = (uchar)val[2];
                }
#endif
            }
        }
    }
}

void writeRectBB(Rect& rect_bb, const char* filename){
    fstream fs(filename, ios::out);
    fs << rect_bb.x << " " << rect_bb.y << " " << rect_bb.width << " " << rect_bb.height << endl;
}

void writeRectMat(Mat& rect_mat, Rect& rect_bb, const char* filename){
    fstream fs(filename, ios::out);
    fs << rect_mat.rows << " " << rect_mat.cols<<endl;
    int r1 = rect_bb.y;
    int r2 = rect_bb.y + rect_bb.height;
    int c1 = rect_bb.x;
    int c2 = rect_bb.x + rect_bb.width;
    for(int r = r1; r <= r2; ++r){
        for(int c = c1; c <= c2; ++c){
            fs << rect_mat.at<Vec2d>(r, c)[0]<<" " << rect_mat.at<Vec2d>(r, c)[1]<<" ";
        }
    }
}

void readRectBB(Rect& rect_bb, const char* filename){
    fstream fs(filename, ios::in);
    fs >> rect_bb.x >> rect_bb.y >> rect_bb.width >> rect_bb.height;
}

void readRectMat(Mat& rect_mat, Rect& rect_bb, const char* filename){
    fstream fs(filename, ios::in);
    int w, h;
    fs >> h >> w;

    rect_mat = -Mat::ones(h, w, CV_64FC2);
    int r1 = rect_bb.y;
    int r2 = rect_bb.y + rect_bb.height;
    int c1 = rect_bb.x;
    int c2 = rect_bb.x + rect_bb.width;
    for(int r = r1; r <= r2; ++r){
        for(int c = c1; c <= c2; ++c){
            fs >> rect_mat.at<Vec2d>(r, c)[0]>> rect_mat.at<Vec2d>(r, c)[1];
        }
    }
}

#if 0
void readRectData(vector<Mat>& rect_masks, vector<Mat>& fish_masks, vector<Rect>& rect_bbs,
                  string rect_mask_dir, string fish_mask_dir, string rect_bbs_dir){
    vector<string> equi_mask_files;
    vector<string> fish_mask_files;
    vector<string> rect_bb_files;

    getFilenames(rect_bbs_dir, rect_bb_files);
    getFilenames(rect_mask_dir, equi_mask_files);
    getFilenames(fish_mask_dir, fish_mask_files);
    rect_masks.resize(rect_bb_files.size());
    fish_masks.resize(rect_bb_files.size());
    rect_bbs.resize(rect_bb_files.size());
    for(size_t i = 0; i < rect_bb_files.size(); ++i){
        readRectBB(rect_bbs[i], rect_bb_files[i].c_str());
        rect_masks[i] = imread(equi_mask_files[i].c_str(), cv::IMREAD_UNCHANGED);
        fish_masks[i] = imread(fish_mask_files[i].c_str(), cv::IMREAD_UNCHANGED);
    }
}
#endif

void sphereImage(Mat& img, Mat& img_sph, Mat& R, Mat& K, Mat& D){
    //int w = img.cols;
    //int h = img.rows;
    int w = img.rows;
    int h = img.cols;
    img_sph = Mat::zeros(h, w, CV_8UC3);
    Mat R_inv = R.t();
    for(int r = 0; r < h; ++r){
        for(int c = 0; c < w; ++c){
            // r : azimuth sampling, c: co-latitude sampling
            // (r, c) --> (azimuth (phi),  co-latitude (theta))
            double theta = (double)c / (double) w * CV_PI/2; // (0-PI/2)
            double phi = (double)r / (double) h * 2 * CV_PI; // (0- 2 * PI)
            // (longitude, latitude) --> spherical coordinates
            double xs = cos(phi) * sin(theta);
            double ys = sin(phi) * sin(theta);
            double zs = cos(theta);

            // spherical coordinates --> fisheye image coordinates
            Mat ps = Mat::zeros(1, 1, CV_64FC3);
            ps.at<Vec3d>(0) = Vec3d(xs, ys, zs);
            // spherical coordinates --> rotate points
            Mat ps_vec = Mat::zeros(3, 1, CV_64FC1);
            ps_vec.at<double>(0) = xs;
            ps_vec.at<double>(1) = ys;
            ps_vec.at<double>(2) = zs;

            Mat pr_vec = R * ps_vec;
            Mat pr = Mat::zeros(1, 1, CV_64FC3);
            pr.at<Vec3d>(0)[0] = pr_vec.at<double>(0);
            pr.at<Vec3d>(0)[1] = pr_vec.at<double>(1);
            pr.at<Vec3d>(0)[2] = pr_vec.at<double>(2);
            Mat pf = Mat::zeros(1, 1, CV_64FC2);
            cv::widefov::projectPoint(pr, pf, K, D, K.at<double>(0,1) / K.at<double>(0, 0));
            int xf = cvRound(pf.at<Vec2d>(0)[0]);
            int yf = cvRound(pf.at<Vec2d>(0)[1]);
            if(xf < img.cols && xf >= 0 && yf < img.rows && yf >= 0){
                //if(pr.at<double>(2)>=0.0)
                    img_sph.at<Vec3b>(r, c) = img.at<Vec3b>(yf, xf);
            }
        }
    }
}

}} // namespace end scope
