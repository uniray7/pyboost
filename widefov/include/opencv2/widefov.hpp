#ifndef __OPENCV_WIDEFOV_HPP__
#define __OPENCV_WIDEFOV_HPP__

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"

#include <vector>
#include <string>

/** @defgroup widefov Wide-angle FOV fisheye camera model
 * 
 * widefov-fisheye model = fisheye model that can handle fov > 180 degrees
*/

namespace cv
{
namespace widefov
{

//! @addtogroup widefov
//! @{

    enum{
        CALIB_USE_INTRINSIC_GUESS   = 1,
        CALIB_RECOMPUTE_EXTRINSIC   = 2,
        CALIB_CHECK_COND            = 4,
        CALIB_FIX_SKEW              = 8,
        CALIB_FIX_K1                = 16,
        CALIB_FIX_K2                = 32,
        CALIB_FIX_K3                = 64,
        CALIB_FIX_K4                = 128,
        CALIB_FIX_INTRINSIC         = 256
    };

    /** @brief Projects points using widefov-fisheye model

    @param objectPoints Array of object points, 1xN/Nx1 3-channel (or vector\<Point3f\> ), where N is
    the number of points in the view.
    @param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or
    vector\<Point2f\>.
    @param affine
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param alpha The skew coefficient.
    @param jacobian Optional output 2Nx15 jacobian matrix of derivatives of image points with respect
    to components of the focal lengths, coordinates of the principal point, distortion coefficients,
    rotation vector, translation vector, and the skew. In the old interface different components of
    the jacobian are returned via different output parameters.

    The function computes projections of 3D points to the image plane given intrinsic and extrinsic
    camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
    image points coordinates (as functions of all the input parameters) with respect to the particular
    parameters, intrinsic and/or extrinsic.
     */
    CV_EXPORTS void projectPoints(InputArray objectPoints, OutputArray imagePoints, const Affine3d& affine,
        InputArray K, InputArray D, double alpha = 0, OutputArray jacobian = noArray());

    /** @overload */
    CV_EXPORTS_W void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec,
        InputArray K, InputArray D, double alpha = 0, OutputArray jacobian = noArray());

    /** @brief Projects single point using widefov-fisheye model

    @param objectPoint 3-channel or 1x3/3x1 Matrix represents the object point
    @param imagePoint 2-channel or 1x2/2x1 Matrix represents the image point
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param alpha The skew coefficient.
     */
    CV_EXPORTS_W void projectPoint(InputArray objectPoint, OutputArray imagePoint, InputArray K, InputArray D, double alpha = 0);

    /** @brief Distorts 2D points using widefov-fisheye model.

    @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is
    the number of points in the view.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param alpha The skew coefficient.
    @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
     */
    CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double alpha = 0);

    /** @brief Undistorts 2D points using widefov-fisheye model

    @param distorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is the
    number of points in the view.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
     */
    CV_EXPORTS_W void undistortPoints(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray R = noArray(), InputArray P  = noArray());

    /** @brief Computes undistortion and rectification maps for image transform by cv::remap(). If D is empty zero
    distortion is used, if R or P is empty identity matrixes are used.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
    for details.
    @param map1 The first output map.
    @param map2 The second output map.
     */
    CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, InputArray R, InputArray P,
        const cv::Size& size, int m1type, OutputArray map1, OutputArray map2);

    /** @brief Transforms an image to compensate for widefov-fisheye lens distortion.

    @param distorted image with fisheye lens distortion.
    @param undistorted Output image with compensated fisheye lens distortion.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param Knew Camera matrix of the distorted image. By default, it is the identity matrix but you
    may additionally scale and shift the result by using a different matrix.
    @param new_size

    The function transforms an image to compensate radial and tangential lens distortion.

    The function is simply a combination of widefov::initUndistortRectifyMap (with unity R ) and remap
    (with bilinear interpolation). See the former function for details of the transformation being
    performed.

    See below the results of undistortImage.
       -   a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
            k_4, k_5, k_6) of distortion were optimized under calibration)
        -   b\) result of widefov::undistortImage of widefov-fisheye camera model (all possible coefficients (k_1, k_2,
            k_3, k_4) of fisheye distortion were optimized under calibration)
        -   c\) original image was captured with fisheye lens

    Pictures a) and b) almost the same. But if we consider points of image located far from the center
    of image, we can notice that on image a) these points are distorted.

    ![image](pics/fisheye_undistorted.jpg)
     */
    CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray Knew = cv::noArray(), const Size& new_size = Size());

    /** @brief Estimates new camera matrix for undistortion or rectification.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param image_size
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param balance Sets the new focal length in range between the min focal length and the max focal
    length. Balance is in range of [0, 1].
    @param new_size
    @param fov_scale Divisor for new focal length.
     */
    CV_EXPORTS_W void estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, const Size &image_size, InputArray R,
        OutputArray P, double balance = 0.0, const Size& new_size = Size(), double fov_scale = 1.0);

    /** @brief Performs camera calibaration

    @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
    coordinate space.
    @param imagePoints vector of vectors of the projections of calibration pattern points.
    imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
    objectPoints[i].size() for each i.
    @param image_size Size of the image used only to initialize the intrinsic camera matrix.
    @param K Output 3x3 floating-point camera matrix
    \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If
    widefov::CALIB_USE_INTRINSIC_GUESS/ is specified, some or all of fx, fy, cx, cy must be
    initialized before calling the function.
    @param D Output vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
    That is, each k-th rotation vector together with the corresponding k-th translation vector (see
    the next output parameter description) brings the calibration pattern from the model coordinate
    space (in which object points are specified) to the world coordinate space, that is, a real
    position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
    @param tvecs Output vector of translation vectors estimated for each pattern view.
    @param flags Different flags that may be zero or a combination of the following values:
    -   **widefov::CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center ( imageSize is used), and focal distances are computed in a least-squares fashion.
    -   **widefov::CALIB_RECOMPUTE_EXTRINSIC** Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -   **widefov::CALIB_CHECK_COND** The functions will check validity of condition number.
    -   **widefov::CALIB_FIX_SKEW** Skew coefficient (alpha) is set to zero and stay zero.
    -   **widefov::CALIB_FIX_K1..4** Selected distortion coefficients are set to zeros and stay
    zero.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, const Size& image_size,
        InputOutputArray K, InputOutputArray D, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0,
            TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

    /** @brief Stereo rectification for widefov-fisheye camera model

    @param K1 First camera matrix.
    @param D1 First camera distortion parameters.
    @param K2 Second camera matrix.
    @param D2 Second camera distortion parameters.
    @param imageSize Size of the image used for stereo calibration.
    @param R Rotation matrix between the coordinate systems of the first and the second
    cameras.
    @param tvec Translation vector between coordinate systems of the cameras.
    @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
    camera.
    @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
    camera.
    @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
    @param flags Operation flags that may be zero or CV_CALIB_ZERO_DISPARITY . If the flag is set,
    the function makes the principal points of each camera have the same pixel coordinates in the
    rectified views. And if the flag is not set, the function may still shift the images in the
    horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
    useful image area.
    @param newImageSize New image resolution after rectification. The same size should be passed to
    initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
    is passed (default), it is set to the original imageSize . Setting it to larger value can help you
    preserve details in the original image, especially when there is a big radial distortion.
    @param balance Sets the new focal length in range between the min focal length and the max focal
    length. Balance is in range of [0, 1].
    @param fov_scale Divisor for new focal length.
     */
    CV_EXPORTS_W void stereoRectify(InputArray K1, InputArray D1, InputArray K2, InputArray D2, const Size &imageSize, InputArray R, InputArray tvec,
        OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags, const Size &newImageSize = Size(),
        double balance = 0.0, double fov_scale = 1.0);

    /** @brief Performs stereo calibration

    @param objectPoints Vector of vectors of the calibration pattern points.
    @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
    observed by the first camera.
    @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
    observed by the second camera.
    @param K1 Input/output first camera matrix:
    \f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
    any of widefov::CALIB_USE_INTRINSIC_GUESS , widefov::CV_CALIB_FIX_INTRINSIC are specified,
    some or all of the matrix components must be initialized.
    @param D1 Input/output vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$ of 4 elements.
    @param K2 Input/output second camera matrix. The parameter is similar to K1 .
    @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
    similar to D1 .
    @param imageSize Size of the image used only to initialize intrinsic camera matrix.
    @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
    @param T Output translation vector between the coordinate systems of the cameras.
    @param flags Different flags that may be zero or a combination of the following values:
    -   **widefov::CV_CALIB_FIX_INTRINSIC** Fix K1, K2? and D1, D2? so that only R, T matrices
    are estimated.
    -   **widefov::CALIB_USE_INTRINSIC_GUESS** K1, K2 contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center (imageSize is used), and focal distances are computed in a least-squares fashion.
    -   **widefov::CALIB_RECOMPUTE_EXTRINSIC** Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -   **widefov::CALIB_CHECK_COND** The functions will check validity of condition number.
    -   **widefov::CALIB_FIX_SKEW** Skew coefficient (alpha) is set to zero and stay zero.
    -   **widefov::CALIB_FIX_K1..4** Selected distortion coefficients are set to zeros and stay
    zero.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                  InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                  OutputArray R, OutputArray T, int flags = widefov::CALIB_FIX_INTRINSIC,
                                  TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));
                                  
    /** @brief Performs multi-chessboard stereo calibration
     * 
    @param objectPoints Vector of vectors of the calibration pattern points.
    @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
    observed by the first camera.
    @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
    observed by the second camera.
    @param K1 Input/output first camera matrix:
    \f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
    any of widefov::CALIB_USE_INTRINSIC_GUESS , widefov::CV_CALIB_FIX_INTRINSIC are specified,
    some or all of the matrix components must be initialized.
    @param D1 Input/output vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$ of 4 elements.
    @param K2 Input/output second camera matrix. The parameter is similar to K1 .
    @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
    similar to D1 .
    @param imageSize Size of the image used only to initialize intrinsic camera matrix.
    @param Rckk1 Output vector of Rodrigues rotation vectors estimated for each pattern view, observed by the first camera.
    @param Tckk1 Output vector of translation vectors estimated for each pattern view, observed by the first camera.
    @param Rckk2 Output vector of Rodrigues rotation vectors estimated for each pattern view, observed by the second camera.
    @param Tckk2 Output vector of translation vectors estimated for each pattern view, observed by the second camera.
    @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
    @param T Output translation vector between the coordinate systems of the cameras.
    @param flags Different flags that may be zero or a combination of the following values:
    -   **widefov::CV_CALIB_FIX_INTRINSIC** Fix K1, K2? and D1, D2? so that only R, T matrices
    are estimated.
    -   **widefov::CALIB_USE_INTRINSIC_GUESS** K1, K2 contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center (imageSize is used), and focal distances are computed in a least-squares fashion.
    -   **widefov::CALIB_RECOMPUTE_EXTRINSIC** Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -   **widefov::CALIB_CHECK_COND** The functions will check validity of condition number.
    -   **widefov::CALIB_FIX_SKEW** Skew coefficient (alpha) is set to zero and stay zero.
    -   **widefov::CALIB_FIX_K1..4** Selected distortion coefficients are set to zeros and stay
    zero.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double multiChkBoardStereoCalibrate(InputArrayOfArrays objectPoints, 
        InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
        InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2,
        InputOutputArrayOfArrays Rckk1, InputOutputArrayOfArrays Tckk1, InputOutputArrayOfArrays Rckk2, InputOutputArrayOfArrays Tckk2,
        OutputArray R, OutputArray T, int flags, TermCriteria criteria);
        
    // This function is better move to utils/ImageRemapping
    CV_EXPORTS void remapImage(cv::Mat& img, cv::Mat& img_remap, cv::Mat& remap_mat);
    CV_EXPORTS void rectBoundingBox(cv::Mat& rect_mat, cv::Rect& rect_bb);
    CV_EXPORTS void equirect2Fisheye(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& rect_mat, 
                                     double fov, cv::Mat& mask, cv::Mat& mask_equi);
    CV_EXPORTS void equirect2Fisheye(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& rect_mat, 
                                     cv::Mat& mask, cv::Mat& mask_equi);
    CV_EXPORTS void equirect2FisheyeWithBB(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& maps, 
                                           cv::Mat& mask_equi, cv::Rect& rect_bb, cv::Size img_size);
    CV_EXPORTS void fisheye2Equirect(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& maps, cv::Mat& mask);
    CV_EXPORTS void fisheye2EquirectWithBB(cv::Mat& R, cv::Mat& K, cv::Mat& D, cv::Mat& maps, cv::Mat& mask, cv::Rect& bb);

    CV_EXPORTS void overlapMask(cv::Mat& mask_left, cv::Mat& mask_right, cv::Size mask_size,
                                cv::Mat& K1, cv::Mat& K2, cv::Mat& D1, cv::Mat& D2, cv::Mat& R, double fov);
    CV_EXPORTS void sphereRectify(cv::Mat& E, cv::Mat& R, cv::Mat& R1, cv::Mat& R2);
    CV_EXPORTS void genPixelRay(cv::Mat& imagePoints, cv::Mat& rays, cv::Mat& K, cv::Mat& D);
    
    /** @brief apply stereo rectification to left/right images  */
    CV_EXPORTS void stereoRectify(std::string image_file_left, std::string image_file_right, 
                       std::string rect_file_left, std::string rect_file_right,
                       cv::Mat& R, cv::Mat& T, cv::Mat& K1, cv::Mat& K2, cv::Mat& D1, cv::Mat& D2,
                       cv::Mat& mask_left_equi, cv::Mat& mask_right_equi, cv::Rect& rect_bb_1, cv::Rect& rect_bb_2, double fov);
                       
    CV_EXPORTS void writeRectBB(cv::Rect& rect_bb, const char* filename);
    CV_EXPORTS void writeRectMat(cv::Mat& rect_mat, cv::Rect& rect_bb, const char* filename);
    CV_EXPORTS void readRectBB(cv::Rect& rect_bb, const char* filename);
    CV_EXPORTS void readRectMat(cv::Mat& rect_mat, cv::Rect& rect_bb, const char* filename);
    CV_EXPORTS void computeEssentialMatrix(cv::Mat& R, cv::Mat& T, cv::Mat& E);
    
    CV_EXPORTS void sphericalRectifyE(cv::Mat& E, cv::Mat& R, cv::Mat& R1, cv::Mat& R2);
    CV_EXPORTS void sphericalRectifyRT(cv::Mat& R, cv::Mat& T, cv::Mat& R1, cv::Mat& R2);
    CV_EXPORTS void sphericalRectifyRT(cv::Mat& R, cv::Mat& T, cv::Mat& R1, cv::Mat& R2, cv::Vec2d& theta_range, cv::Vec2d& phi_range);
    CV_EXPORTS void sphereImage(cv::Mat& img, cv::Mat& img_sph, cv::Mat& R, cv::Mat& K, cv::Mat& D);

//! @}

}} // cv::widefov namespace scope end

#endif
