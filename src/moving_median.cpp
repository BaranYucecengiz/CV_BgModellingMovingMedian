#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

void find_median(std::vector<uchar>& pixels, int num_frames, uchar& median) {
    std::nth_element(pixels.begin(), pixels.begin() + num_frames / 2, pixels.end());
    median = pixels[num_frames / 2];
}

cv::Mat calculate_median_background(const std::deque<cv::Mat>& frames) {
    int rows = frames[0].rows;
    int cols = frames[0].cols;
    int num_frames = frames.size();
    cv::Mat median_background(rows, cols, CV_8UC1);

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::vector<uchar> pixels(num_frames);
            for (int j = 0; j < num_frames; ++j) {
                pixels[j] = frames[j].at<uchar>(row, col);
            }
            uchar median;
            find_median(pixels, num_frames, median);
            median_background.at<uchar>(row, col) = median;
        }
    }

    return median_background;
}

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Could not open the webcam." << std::endl;
        return -1;
    }

    omp_set_num_threads(omp_get_max_threads());

    std::deque<cv::Mat> frame_queue;
    int max_frames = 5;

    while (true) {
        cv::Mat frame, gray_frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        frame_queue.push_back(gray_frame.clone());
        if (frame_queue.size() > max_frames) {
            frame_queue.pop_front();
        }

        if (frame_queue.size() == max_frames) {
            cv::Mat median_background = calculate_median_background(frame_queue);

            cv::Mat foreground_mask;
            cv::absdiff(gray_frame, median_background, foreground_mask);
            cv::threshold(foreground_mask, foreground_mask, 30, 255, cv::THRESH_BINARY);

            cv::imshow("Original Frame", frame);
            cv::imshow("Median Background Model", median_background);
            cv::imshow("Foreground Mask", foreground_mask);
        }

        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
