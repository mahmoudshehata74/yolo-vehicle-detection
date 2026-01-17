#include "utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// ------------------------------------------------------------------
//                         FPS Counter
// ------------------------------------------------------------------

FPSCounter::FPSCounter()
	: instantFPS(0.0), averageFPS(0.0), initialized(false) {
}

void FPSCounter::update() {
	auto now = std::chrono::high_resolution_clock::now();

	if (initialized) {
		double frameTime = std::chrono::duration<double>(now - lastTime).count();
		instantFPS = 1.0 / frameTime;

		// Exponential moving average
		if (averageFPS < 1.0)
			averageFPS = instantFPS;
		else
			averageFPS = 0.9 * averageFPS + 0.1 * instantFPS;
	}

	lastTime = now;
	initialized = true;
}

double FPSCounter::getInstantFPS() const { return instantFPS; }
double FPSCounter::getAverageFPS() const { return averageFPS; }

std::string FPSCounter::toString() const {
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "FPS: %.1f (Avg: %.1f)", instantFPS, averageFPS);
	return std::string(buffer);
}

// ------------------------------------------------------------------
//                        Preprocessing
// ------------------------------------------------------------------

cv::Mat letterbox(const cv::Mat& image, LetterboxInfo& info) {
	// Calculate scale to fit image in input size
	float scaleX = (float)Config::INPUT_WIDTH / image.cols;
	float scaleY = (float)Config::INPUT_HEIGHT / image.rows;
	info.scale = std::min(scaleX, scaleY);

	// Calculate new dimensions
	int newW = static_cast<int>(image.cols * info.scale);
	int newH = static_cast<int>(image.rows * info.scale);

	// Calculate padding
	info.padX = (Config::INPUT_WIDTH - newW) / 2;
	info.padY = (Config::INPUT_HEIGHT - newH) / 2;

	// Resize and add padding
	cv::Mat resized, padded;
	cv::resize(image, resized, cv::Size(newW, newH));
	cv::copyMakeBorder(resized, padded,
		info.padY, Config::INPUT_HEIGHT - newH - info.padY,
		info.padX, Config::INPUT_WIDTH - newW - info.padX,
		cv::BORDER_CONSTANT, Config::LETTERBOX_COLOR);

	return padded;
}

std::vector<float> imageToTensor(const cv::Mat& image) {
	// Convert BGR to RGB and normalize
	cv::Mat rgb, normalized;
	cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
	rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);

	// Split channels and convert to CHW format
	std::vector<cv::Mat> channels(3);
	cv::split(normalized, channels);

	std::vector<float> tensor;
	tensor.reserve(3 * Config::INPUT_WIDTH * Config::INPUT_HEIGHT);

	for (int c = 0; c < 3; c++) {
		tensor.insert(tensor.end(),
			(float*)channels[c].datastart,
			(float*)channels[c].dataend);
	}

	return tensor;
}

// ------------------------------------------------------------------
//                        Postprocessing
// ------------------------------------------------------------------

bool isVehicleClass(int classId) {
	auto& classes = Config::VEHICLE_CLASSES;
	return std::find(classes.begin(), classes.end(), classId) != classes.end();
}

bool isSegmentClass(int classId) {
	auto& classes = Config::SEGMENT_CLASSES;
	return std::find(classes.begin(), classes.end(), classId) != classes.end();
}

std::string getClassName(int classId) {
	switch (classId) {
	case 0:  return "Person";
	case 2:  return "Car";
	case 5:  return "Bus";
	case 7:  return "Truck";
	default: return "Object";
	}
}

cv::Scalar getClassColor(int classId) {
	switch (classId) {
	case 0:  return Config::COLOR_PERSON;
	case 2:  return Config::COLOR_CAR;
	case 5:  return Config::COLOR_BUS;
	case 7:  return Config::COLOR_TRUCK;
	default: return cv::Scalar(255, 255, 0);
	}
}

cv::Rect scaleBox(float cx, float cy, float w, float h,
	const LetterboxInfo& info,
	const cv::Size& originalSize) {
	// Remove padding and scale
	cx = (cx - info.padX) / info.scale;
	cy = (cy - info.padY) / info.scale;
	w = w / info.scale;
	h = h / info.scale;

	// Convert from center format to corner format
	int x = static_cast<int>(cx - w / 2);
	int y = static_cast<int>(cy - h / 2);
	int bw = static_cast<int>(w);
	int bh = static_cast<int>(h);

	// Clamp to image bounds
	x = std::max(0, x);
	y = std::max(0, y);
	bw = std::min(bw, originalSize.width - x);
	bh = std::min(bh, originalSize.height - y);

	return cv::Rect(x, y, bw, bh);
}

std::vector<Detection> applyNMS(std::vector<Detection>& detections) {
	if (detections.empty()) return {};

	// Extract boxes and scores
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	for (const auto& det : detections) {
		boxes.push_back(det.box);
		scores.push_back(det.confidence);
	}

	// Apply NMS
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores,
		Config::CONF_THRESHOLD,
		Config::NMS_THRESHOLD,
		indices);

	// Return filtered detections
	std::vector<Detection> result;
	result.reserve(indices.size());
	for (int idx : indices) {
		result.push_back(detections[idx]);
	}

	return result;
}

std::vector<Segmentation> applyNMSSeg(std::vector<Segmentation>& segmentations) {
	if (segmentations.empty()) return {};

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	for (const auto& seg : segmentations) {
		boxes.push_back(seg.box);
		scores.push_back(seg.confidence);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores,
		Config::CONF_THRESHOLD,
		Config::NMS_THRESHOLD,
		indices);

	std::vector<Segmentation> result;
	result.reserve(indices.size());
	for (int idx : indices) {
		result.push_back(segmentations[idx]);
	}

	return result;
}

// ------------------------------------------------------------------
//                        Mask Processing
// ------------------------------------------------------------------

cv::Mat processMask(const std::vector<float>& maskCoeffs,
	const float* maskPrototypes,
	const cv::Rect& box,
	const cv::Size& originalSize,
	const LetterboxInfo& info) {

	// Create coefficient matrix [1, 32]
	cv::Mat coeffsMat(1, Config::SEG_CHANNELS, CV_32F);
	for (int i = 0; i < Config::SEG_CHANNELS; i++) {
		coeffsMat.at<float>(0, i) = maskCoeffs[i];
	}

	// Create prototype matrix [32, 160*160]
	cv::Mat protoMat(Config::SEG_CHANNELS,
		Config::SEG_HEIGHT * Config::SEG_WIDTH,
		CV_32F,
		(void*)maskPrototypes);

	// Matrix multiplication: [1, 32] x [32, 25600] = [1, 25600]
	cv::Mat maskFlat = coeffsMat * protoMat;
	cv::Mat mask = maskFlat.reshape(1, Config::SEG_HEIGHT);

	// Apply sigmoid activation
	cv::exp(-mask, mask);
	mask = 1.0f / (1.0f + mask);

	// Resize to input size
	cv::Mat maskResized;
	cv::resize(mask, maskResized, cv::Size(Config::INPUT_WIDTH, Config::INPUT_HEIGHT),
		0, 0, cv::INTER_LINEAR);

	// Calculate box coordinates in letterboxed image
	int x1 = static_cast<int>(box.x * info.scale + info.padX);
	int y1 = static_cast<int>(box.y * info.scale + info.padY);
	int x2 = static_cast<int>((box.x + box.width) * info.scale + info.padX);
	int y2 = static_cast<int>((box.y + box.height) * info.scale + info.padY);

	// Clamp coordinates
	x1 = std::max(0, std::min(x1, Config::INPUT_WIDTH - 1));
	y1 = std::max(0, std::min(y1, Config::INPUT_HEIGHT - 1));
	x2 = std::max(x1 + 1, std::min(x2, Config::INPUT_WIDTH));
	y2 = std::max(y1 + 1, std::min(y2, Config::INPUT_HEIGHT));

	// Crop mask to bounding box
	cv::Mat croppedMask = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_32F);
	if (x2 > x1 && y2 > y1) {
		maskResized(cv::Rect(x1, y1, x2 - x1, y2 - y1))
			.copyTo(croppedMask(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
	}

	// Remove letterbox padding
	int cropX = std::max(0, info.padX);
	int cropY = std::max(0, info.padY);
	int cropW = std::max(1, std::min(Config::INPUT_WIDTH - 2 * info.padX,
		Config::INPUT_WIDTH - cropX));
	int cropH = std::max(1, std::min(Config::INPUT_HEIGHT - 2 * info.padY,
		Config::INPUT_HEIGHT - cropY));

	cv::Mat maskCropped = croppedMask(cv::Rect(cropX, cropY, cropW, cropH));

	// Resize to original image size
	cv::Mat maskFinal;
	cv::resize(maskCropped, maskFinal, originalSize, 0, 0, cv::INTER_LINEAR);

	// Convert to binary mask
	cv::Mat binaryMask;
	cv::threshold(maskFinal, binaryMask, Config::MASK_THRESHOLD, 255, cv::THRESH_BINARY);
	binaryMask.convertTo(binaryMask, CV_8U);

	return binaryMask;
}

// ------------------------------------------------------------------
//                          Drawing
// ------------------------------------------------------------------

void drawDetection(cv::Mat& image, const Detection& det) {
	// Draw bounding box
	cv::rectangle(image, det.box, det.color, 2);

	// Create label text
	char label[64];
	snprintf(label, sizeof(label), "%s %.0f%%", det.className.c_str(), det.confidence * 100);

	// Calculate label position
	int baseline;
	cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
	int top = std::max(det.box.y, textSize.height + 10);

	// Draw label background
	cv::rectangle(image,
		cv::Point(det.box.x, top - textSize.height - 10),
		cv::Point(det.box.x + textSize.width + 5, top),
		det.color, cv::FILLED);

	// Draw label text
	cv::putText(image, label, cv::Point(det.box.x + 2, top - 5),
		cv::FONT_HERSHEY_SIMPLEX, 0.6, Config::COLOR_WHITE, 2);
}

void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
	for (const auto& det : detections) {
		drawDetection(image, det);
	}
}

void drawSegmentation(cv::Mat& image, const Segmentation& seg, float alpha) {
	// Draw mask overlay
	if (!seg.mask.empty()) {
		cv::Mat colorMask = cv::Mat::zeros(image.size(), image.type());
		colorMask.setTo(seg.color, seg.mask);
		cv::addWeighted(image, 1.0, colorMask, alpha, 0, image);
	}

	// Draw bounding box
	cv::rectangle(image, seg.box, seg.color, 2);

	// Create and draw label
	char label[64];
	snprintf(label, sizeof(label), "%s %.0f%%", seg.className.c_str(), seg.confidence * 100);

	int baseline;
	cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
	int top = std::max(seg.box.y, textSize.height + 10);

	cv::rectangle(image,
		cv::Point(seg.box.x, top - textSize.height - 10),
		cv::Point(seg.box.x + textSize.width + 5, top),
		seg.color, cv::FILLED);

	cv::putText(image, label, cv::Point(seg.box.x + 2, top - 5),
		cv::FONT_HERSHEY_SIMPLEX, 0.6, Config::COLOR_WHITE, 2);
}

void drawSegmentations(cv::Mat& image, const std::vector<Segmentation>& segmentations, float alpha) {
	for (const auto& seg : segmentations) {
		drawSegmentation(image, seg, alpha);
	}
}

void drawFPS(cv::Mat& image, const FPSCounter& fps, int numDetections) {
	// Background
	cv::rectangle(image, cv::Point(10, 10), cv::Point(350, 80),
		Config::COLOR_BLACK, cv::FILLED);

	// FPS text
	cv::putText(image, fps.toString(), cv::Point(20, 40),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, Config::COLOR_FPS, 2);

	// Detections count
	char text[32];
	snprintf(text, sizeof(text), "Detections: %d", numDetections);
	cv::putText(image, text, cv::Point(20, 70),
		cv::FONT_HERSHEY_SIMPLEX, 0.7, Config::COLOR_WHITE, 2);
}

void drawInfo(cv::Mat& image, const FPSCounter& fps, int numDetections, float alpha) {
	// Background
	cv::rectangle(image, cv::Point(10, 10), cv::Point(380, 110),
		Config::COLOR_BLACK, cv::FILLED);

	// FPS
	cv::putText(image, fps.toString(), cv::Point(20, 40),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, Config::COLOR_FPS, 2);

	// Detections
	char detText[32];
	snprintf(detText, sizeof(detText), "Detections: %d", numDetections);
	cv::putText(image, detText, cv::Point(20, 70),
		cv::FONT_HERSHEY_SIMPLEX, 0.7, Config::COLOR_WHITE, 2);

	// Transparency
	char alphaText[32];
	snprintf(alphaText, sizeof(alphaText), "Transparency: %.0f%%", alpha * 100);
	cv::putText(image, alphaText, cv::Point(20, 100),
		cv::FONT_HERSHEY_SIMPLEX, 0.7, Config::COLOR_WHITE, 2);
}

// ------------------------------------------------------------------
//                          Utilities
// ------------------------------------------------------------------

bool openVideoSource(cv::VideoCapture& cap, const std::string& source) {
	if (source == "0" || source == "camera") {
		cap.open(0);
	}
	else if (source.substr(0, 7) == "camera:") {
		int camIndex = std::stoi(source.substr(7));
		cap.open(camIndex);
	}
	else {
		cap.open(source);
	}
	return cap.isOpened();
}

void printUsage(const char* programName) {
	std::cout << "\n";
	std::cout << "========================================\n";
	std::cout << "     YOLO Vehicle Detector (Task 1)     \n";
	std::cout << "========================================\n";
	std::cout << "\nUsage: " << programName << " <input>\n\n";
	std::cout << "Input:\n";
	std::cout << "  video.mp4        Video file\n";
	std::cout << "  camera           Default camera\n";
	std::cout << "  camera:1         Specific camera\n";
	std::cout << "  rtsp://...       RTSP stream\n";
	std::cout << "\nControls:\n";
	std::cout << "  ESC/Q - Quit\n";
	std::cout << "  S     - Screenshot\n";
	std::cout << "  R     - Toggle recording\n\n";
}

void printSegmenterUsage(const char* programName) {
	std::cout << "\n";
	std::cout << "========================================\n";
	std::cout << "    YOLO Vehicle Segmenter (Task 2)     \n";
	std::cout << "========================================\n";
	std::cout << "\nUsage: " << programName << " <input>\n\n";
	std::cout << "Input:\n";
	std::cout << "  video.mp4        Video file\n";
	std::cout << "  camera           Default camera\n";
	std::cout << "  camera:1         Specific camera\n";
	std::cout << "  rtsp://...       RTSP stream\n";
	std::cout << "\nControls:\n";
	std::cout << "  ESC/Q - Quit\n";
	std::cout << "  +/-   - Adjust transparency\n";
	std::cout << "  S     - Screenshot\n";
	std::cout << "  R     - Toggle recording\n\n";
}

void printStartupInfo(const cv::VideoCapture& cap, const std::string& source) {
	std::cout << "[OK] Source: " << source << "\n";
	std::cout << "     Size: " << cap.get(cv::CAP_PROP_FRAME_WIDTH)
		<< "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
	std::cout << "     FPS: " << cap.get(cv::CAP_PROP_FPS) << "\n";
}