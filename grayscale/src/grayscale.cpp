#include "grayscale.hpp"

namespace gray{
  cv::Mat grayscale(cv::Mat cv_img) {
    cv::Mat gray_result;
    cv::cvtColor(cv_img, gray_result, cv::COLOR_BGR2GRAY);
    return gray_result;
  }
}
