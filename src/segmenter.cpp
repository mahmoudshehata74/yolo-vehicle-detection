#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>


const int INPUT_W = 640 , INPUT_H = 640;
const int NUM_PREDICTIONS = 8400;
const int NUM_CLASSES = 80;
const int NUM_MASKS = 32 , MASK_H = 160 , MASK_W = 160;

const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
const float MASK_THRESHOLD = 0.5f;

const std::vector<int> TARGET_CLASSES = { 0, 2, 5, 7 };


float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

float toProb(float v) {
	if (v < 0.0f || v > 1.0f) return sigmoid(v);
	return v;
}

cv::Scalar getColor(int classId) {
	switch (classId) {
	case 0: return cv::Scalar(255, 0, 255); // person - magenta
	case 2: return cv::Scalar(0, 255, 0);   // car - green
	case 5: return cv::Scalar(255, 165, 0); // bus - orange
	case 7: return cv::Scalar(0, 0, 255);   // truck - red
	default: return cv::Scalar(0, 255, 255);
	}
}

const char* getName(int classId) {
	switch (classId) {
	case 0: return "Person";
	case 2: return "Car";
	case 5: return "Bus";
	case 7: return "Truck";
	default: return "Object";
	}
}

// Build 160x160 mask from prototype and coefficients
cv::Mat buildMask(const float* proto, const std::array<float, 32>& coef) {
	cv::Mat coeffMat(1, NUM_MASKS, CV_32F, (void*)coef.data());
	cv::Mat protoMat(NUM_MASKS, MASK_H * MASK_W, CV_32F, (void*)proto);

	cv::Mat maskFlat = coeffMat * protoMat;
	cv::Mat mask = maskFlat.reshape(1, MASK_H);
	cv::exp(-mask, mask);
	cv::add(1.0f, mask, mask);
	cv::divide(1.0f, mask, mask);

	return mask;
}

// Apply mask overlay on image
void applyMask(cv::Mat& image, const cv::Mat& mask160,
	const cv::Rect& box, const cv::Scalar& color,
	float alpha, float scaleX, float scaleY) {

	float invScaleX = 1.0f / scaleX;
	float invScaleY = 1.0f / scaleY;
	float maskRatio = MASK_W / (float)INPUT_W;
	int mx1 = (int)(box.x * invScaleX * maskRatio);
	int my1 = (int)(box.y * invScaleY * maskRatio);
	int mx2 = (int)((box.x + box.width) * invScaleX * maskRatio);
	int my2 = (int)((box.y + box.height) * invScaleY * maskRatio);

	mx1 = std::max(0, std::min(mx1, MASK_W - 1));
	my1 = std::max(0, std::min(my1, MASK_H - 1));
	mx2 = std::max(mx1 + 1, std::min(mx2, MASK_W));
	my2 = std::max(my1 + 1, std::min(my2, MASK_H));

	if (mx2 <= mx1 || my2 <= my1) return;

	// Crop and resize mask
	cv::Mat maskCrop = mask160(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)).clone();
	cv::Mat maskResized;
	cv::resize(maskCrop, maskResized, cv::Size(box.width, box.height), 0, 0, cv::INTER_LINEAR);
	cv::Mat maskBin;
	cv::threshold(maskResized, maskBin, MASK_THRESHOLD, 255, cv::THRESH_BINARY);
	maskBin.convertTo(maskBin, CV_8U);

	// Get safe ROI
	cv::Rect safeBox = box & cv::Rect(0, 0, image.cols, image.rows);
	if (safeBox.width <= 0 || safeBox.height <= 0) return;

	cv::Mat roi = image(safeBox);
	int dx = safeBox.x - box.x;
	int dy = safeBox.y - box.y;
	cv::Rect maskRect(dx, dy, safeBox.width, safeBox.height);
	maskRect &= cv::Rect(0, 0, maskBin.cols, maskBin.rows);

	if (maskRect.width <= 0 || maskRect.height <= 0) return;

	cv::Mat maskROI = maskBin(maskRect);

	if (maskROI.size() != roi.size()) {
		cv::resize(maskROI, maskROI, roi.size(), 0, 0, cv::INTER_NEAREST);
	}

	cv::Mat colorMask(roi.size(), roi.type(), color);
	cv::Mat blended;
	cv::addWeighted(roi, 1.0 - alpha, colorMask, alpha, 0, blended);
	blended.copyTo(roi, maskROI);
}


int main(int argc, char** argv) {

	if (argc < 2) {
		std::cout << "\n";
		std::cout << "========================================\n";
		std::cout << "    YOLO Vehicle Segmenter      \n";
		std::cout << "========================================\n";
		std::cout << "\nUsage: task2_segmenter <input>\n\n";
		std::cout << "Input:\n";
		std::cout << "  video.mp4    Video file\n";
		std::cout << "  camera       Default camera\n";
		std::cout << "  rtsp://...   RTSP stream\n";
		std::cout << "\nControls:\n";
		std::cout << "  ESC/Q - Quit\n";
		std::cout << "  +/-   - Adjust transparency\n";
		std::cout << "  S     - Screenshot\n";
		std::cout << "  R     - Toggle recording\n\n";
		return 0;
	}

	std::string source = argv[1];

	// Open video source
	cv::VideoCapture cap;
	if (source == "0" || source == "camera")
		cap.open(0);
	else
		cap.open(source);

	if (!cap.isOpened()) {
		std::cerr << "[ERROR] Cannot open: " << source << "\n";
		return 1;
	}

	int frameW = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameH = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	double videoFps = cap.get(cv::CAP_PROP_FPS);
	if (videoFps <= 0) videoFps = 30.0;

	std::cout << "[OK] Source: " << source << "\n";
	std::cout << "     Size: " << frameW << "x" << frameH << "\n";
	std::cout << "     FPS: " << videoFps << "\n";

	cv::VideoWriter writer;
	bool recording = true;
	std::string outputFile = "demo_segmentation.mp4";

	writer.open(outputFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
		videoFps, cv::Size(frameW, frameH));

	if (writer.isOpened())
		std::cout << "[OK] Recording: " << outputFile << "\n";
	else {
		std::cout << "[WARN] Cannot create output file\n";
		recording = false;
	}

	const ORTCHAR_T* modelPath = ORT_TSTR("models/yolo11n-seg.onnx");

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "segmenter");
	Ort::SessionOptions options;

	try {
		OrtCUDAProviderOptions cudaOpts{};
		cudaOpts.device_id = 0;
		options.AppendExecutionProvider_CUDA(cudaOpts);
		std::cout << "[OK] Using CUDA\n";
	}
	catch (...) {
		std::cout << "[WARN] CUDA not available\n";
	}

	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	Ort::Session session(env, modelPath, options);
	std::cout << "[OK] Model loaded\n\n";

	std::string winName = "YOLO Vehicle Segmenter";
	cv::namedWindow(winName, cv::WINDOW_NORMAL);

	std::vector<float> inputData(3 * INPUT_W * INPUT_H);
	float alpha = 0.5f;
	double fps = 0.0;
	double avgFps = 0.0;
	int frameCount = 0;
	int screenshotNum = 0;

	std::cout << "[INFO] Press ESC to quit\n\n";

	cv::Mat frame;

	// Main loop
	while (cap.read(frame)) {
		auto startTime = std::chrono::high_resolution_clock::now();

		cv::Mat img = frame.clone();

		cv::Mat resized, rgb, blob;
		cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
		cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
		rgb.convertTo(blob, CV_32F, 1.0 / 255.0);

		std::vector<cv::Mat> channels(3);
		cv::split(blob, channels);
		int chSize = INPUT_W * INPUT_H;
		for (int c = 0; c < 3; c++) {
			memcpy(inputData.data() + c * chSize, channels[c].data, chSize * sizeof(float));
		}

		std::vector<int64_t> inputShape = { 1, 3, INPUT_H, INPUT_W };
		auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
			memInfo, inputData.data(), inputData.size(),
			inputShape.data(), inputShape.size()
		);

		const char* inNames[] = { "images" };
		const char* outNames[] = { "output0", "output1" };

		auto output = session.Run(Ort::RunOptions{ nullptr },
			inNames, &inputTensor, 1,
			outNames, 2);

		float* detData = output[0].GetTensorMutableData<float>();
		float* maskData = output[1].GetTensorMutableData<float>();

		std::vector<cv::Rect> boxes;
		std::vector<float> scores;
		std::vector<int> classIds;
		std::vector<std::array<float, NUM_MASKS>> maskCoeffs;

		float scaleX = (float)img.cols / INPUT_W;
		float scaleY = (float)img.rows / INPUT_H;

		for (int i = 0; i < NUM_PREDICTIONS; i++) {
			float bestScore = -1.0f;
			int bestClass = -1;

			for (int cid : TARGET_CLASSES) {
				float score = toProb(detData[(4 + cid) * NUM_PREDICTIONS + i]);
				if (score > bestScore) {
					bestScore = score;
					bestClass = cid;
				}
			}

			if (bestScore < CONF_THRESHOLD) continue;

			float cx = detData[0 * NUM_PREDICTIONS + i] * scaleX;
			float cy = detData[1 * NUM_PREDICTIONS + i] * scaleY;
			float w = detData[2 * NUM_PREDICTIONS + i] * scaleX;
			float h = detData[3 * NUM_PREDICTIONS + i] * scaleY;

			int x = (int)(cx - w / 2);
			int y = (int)(cy - h / 2);
			int bw = (int)w;
			int bh = (int)h;

			x = std::max(0, x);
			y = std::max(0, y);
			bw = std::min(bw, img.cols - x);
			bh = std::min(bh, img.rows - y);

			if (bw <= 0 || bh <= 0) continue;

			std::array<float, NUM_MASKS> coef{};
			for (int k = 0; k < NUM_MASKS; k++) {
				coef[k] = detData[(4 + NUM_CLASSES + k) * NUM_PREDICTIONS + i];
			}

			boxes.emplace_back(x, y, bw, bh);
			scores.push_back(bestScore);
			classIds.push_back(bestClass);
			maskCoeffs.push_back(coef);
		}

		// Apply NMS
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);
		for (int idx : indices) {
			cv::Rect box = boxes[idx];
			cv::Scalar color = getColor(classIds[idx]);
			cv::Mat mask = buildMask(maskData, maskCoeffs[idx]);
			applyMask(img, mask, box, color, alpha, scaleX, scaleY);
			cv::rectangle(img, box, color, 2);
			std::string label = cv::format("%s %.0f%%", getName(classIds[idx]), scores[idx] * 100);

			int baseline;
			cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
			int labelY = std::max(box.y, textSize.height + 5);

			cv::rectangle(img,
				cv::Point(box.x, labelY - textSize.height - 5),
				cv::Point(box.x + textSize.width + 5, labelY),
				color, cv::FILLED);

			cv::putText(img, label, cv::Point(box.x + 2, labelY - 3),
				cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
		}

		// Calculate FPS
		auto endTime = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double>(endTime - startTime).count();
		fps = 1.0 / std::max(elapsed, 1e-9);
		avgFps = (avgFps < 1.0) ? fps : (0.9 * avgFps + 0.1 * fps);

		cv::rectangle(img, cv::Point(10, 10), cv::Point(350, 105), cv::Scalar(0, 0, 0), cv::FILLED);

		cv::putText(img, cv::format("FPS: %.1f (Avg: %.1f)", fps, avgFps),
			cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

		cv::putText(img, cv::format("Detections: %d", (int)indices.size()),
			cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

		std::string statusText = recording ? "REC" : cv::format("Alpha: %d%%", (int)(alpha * 100));
		cv::Scalar statusColor = recording ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255);
		cv::putText(img, statusText, cv::Point(20, 85),
			cv::FONT_HERSHEY_SIMPLEX, 0.6, statusColor, 2);

		if (recording && writer.isOpened()) {
			writer.write(img);
		}

		cv::imshow(winName, img);
		frameCount++;

		int key = cv::waitKey(1) & 0xFF;

		if (key == 27 || key == 'q' || key == 'Q')
			break;

		if (key == '+' || key == '=')
			alpha = std::min(1.0f, alpha + 0.1f);

		if (key == '-' || key == '_')
			alpha = std::max(0.1f, alpha - 0.1f);

		if (key == 's' || key == 'S') {
			std::string fname = "screenshot_" + std::to_string(screenshotNum++) + ".jpg";
			cv::imwrite(fname, img);
			std::cout << "[OK] Saved: " << fname << "\n";
		}

		if (key == 'r' || key == 'R') {
			recording = !recording;
			std::cout << "[INFO] Recording: " << (recording ? "ON" : "OFF") << "\n";
		}
	}

	cap.release();
	writer.release();
	cv::destroyAllWindows();

	std::cout << "\n[DONE]\n";
	std::cout << "  Frames: " << frameCount << "\n";
	std::cout << "  Avg FPS: " << (int)avgFps << "\n";
	std::cout << "  Output: " << outputFile << "\n";

	return 0;
}