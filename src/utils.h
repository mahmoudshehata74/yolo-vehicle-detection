#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <chrono>

//                          Constants

namespace Config {
	const int INPUT_WIDTH = 640 , INPUT_HEIGHT = 640 , NUM_CLASSES = 80 , NUM_PREDICTIONS = 8400;

	const int SEG_CHANNELS = 32 , SEG_WIDTH = 160 , SEG_HEIGHT = 160;

	// Thresholds
	const float CONF_THRESHOLD = 0.25f;
	const float NMS_THRESHOLD = 0.45f;
	const float MASK_THRESHOLD = 0.5f;

	const std::vector<int> VEHICLE_CLASSES = { 2, 5, 7 };
	const std::vector<int> SEGMENT_CLASSES = { 0, 2, 5, 7 };

	const cv::Scalar COLOR_PERSON = cv::Scalar(255, 0, 0);
	const cv::Scalar COLOR_CAR = cv::Scalar(0, 255, 0);
	const cv::Scalar COLOR_BUS = cv::Scalar(255, 165, 0);
	const cv::Scalar COLOR_TRUCK = cv::Scalar(0, 0, 255);
	const cv::Scalar COLOR_FPS = cv::Scalar(0, 255, 0);
	const cv::Scalar COLOR_WHITE = cv::Scalar(255, 255, 255);
	const cv::Scalar COLOR_BLACK = cv::Scalar(0, 0, 0);
	const cv::Scalar LETTERBOX_COLOR = cv::Scalar(114, 114, 114);

	// Mask transparency
	const float DEFAULT_ALPHA = 0.5f;
}

//                          Structures

struct Detection {
	cv::Rect box;
	float confidence;
	int classId;
	std::string className;
	cv::Scalar color;
};

struct Segmentation {
	cv::Rect box;
	float confidence;
	int classId;
	std::string className;
	cv::Scalar color;
	cv::Mat mask;
	std::vector<float> maskCoeffs;
};

struct LetterboxInfo {
	float scale;
	int padX;
	int padY;
};

// FPS counter class
class FPSCounter {
private:
	double instantFPS;
	double averageFPS;
	std::chrono::high_resolution_clock::time_point lastTime;
	bool initialized;

public:
	FPSCounter();
	void update();
	double getInstantFPS() const;
	double getAverageFPS() const;
	std::string toString() const;
};

//                       Preprocessing

cv::Mat letterbox(const cv::Mat& image, LetterboxInfo& info);
std::vector<float> imageToTensor(const cv::Mat& image);

//                       Postprocessing

bool isVehicleClass(int classId);
bool isSegmentClass(int classId);
std::string getClassName(int classId);
cv::Scalar getClassColor(int classId);

cv::Rect scaleBox(float cx, float cy, float w, float h,
	const LetterboxInfo& info,
	const cv::Size& originalSize);

std::vector<Detection> applyNMS(std::vector<Detection>& detections);
std::vector<Segmentation> applyNMSSeg(std::vector<Segmentation>& segmentations);

//                       Segmentation

cv::Mat processMask(const std::vector<float>& maskCoeffs,
	const float* maskPrototypes,
	const cv::Rect& box,
	const cv::Size& originalSize,
	const LetterboxInfo& info);

//                         Drawing

void drawDetection(cv::Mat& image, const Detection& det);
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);
void drawSegmentation(cv::Mat& image, const Segmentation& seg, float alpha);
void drawSegmentations(cv::Mat& image, const std::vector<Segmentation>& segmentations, float alpha);
void drawFPS(cv::Mat& image, const FPSCounter& fps, int numDetections);
void drawInfo(cv::Mat& image, const FPSCounter& fps, int numDetections, float alpha);

//                         Utilities

bool openVideoSource(cv::VideoCapture& cap, const std::string& source);
void printUsage(const char* programName);
void printSegmenterUsage(const char* programName);
void printStartupInfo(const cv::VideoCapture& cap, const std::string& source);

#endif // UTILS_H