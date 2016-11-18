#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter.hpp>

// code gen ~
#include <grayscale.hpp>

namespace pbcvt {

  using namespace boost::python;
  using namespace cv;


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
    def("grayscale", gray::grayscale);
  }

} //end namespace pbcvt
