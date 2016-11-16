#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <boost/python.hpp>
#include <pyboostcvconverter.hpp>

#include <iostream>
#include <typeinfo>
#include <string>

namespace pbcvt {

  using namespace boost::python;
  using namespace cv;


  cv::Mat grayscale(cv::Mat cv_img) {
    cv::Mat gray_result;
    cvtColor(cv_img, gray_result, COLOR_BGR2GRAY);
    return gray_result;
  }



  //This example uses Mat directly, but we won't need to worry about the conversion
  /**
   * Example function. Basic inner matrix product using implicit matrix conversion.
   * @param leftMat left-hand matrix operand
   * @param rightMat right-hand matrix operand
   * @return an NdArray representing the dot-product of the left and right operands
   */
  cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
    auto c1 = leftMat.cols, r2 = rightMat.rows;
    if (c1 != r2) {
      PyErr_SetString(PyExc_TypeError,
          "Incompatible sizes for matrix multiplication.");
      throw_error_already_set();
    }
    cv::Mat result = leftMat * rightMat;

    return result;
  }


#if (PY_VERSION_HEX >= 0x03000000)

  static void *init_ar() {
#else
    static void init_ar(){
#endif
      Py_Initialize();

      import_array();
      return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
      //using namespace XM;
      init_ar();

      //initialize converters
      to_python_converter<cv::Mat,
        pbcvt::matToNDArrayBoostConverter>();
      pbcvt::matFromNDArrayBoostConverter();

      //expose module-level functions
      def("dot2", dot2);
      def("grayscale", grayscale);


    }

  } //end namespace pbcvt
