#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>


const int INPUT_W = 640 , INPUT_H = 640 , NUM_PREDICTIONS = 8400, NUM_CLASSES = 80;
const float CONF_THRESHOLD = 0.25f , NMS_THRESHOLD = 0.45f;

// To Detect vehicle classes only: car(2), bus(5), truck(7)
const std::vector<int> VEHICLE_CLASSES = { 2, 5, 7 };


float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

float toProb(float v) {
	if (v < 0.0f || v > 1.0f) return sigmoid(v);
	return v;
}

cv::Scalar getColor(int classId) {
	switch (classId) {
	case 2: return cv::Scalar(0, 255, 0);   // car - green
	case 5: return cv::Scalar(255, 165, 0); // bus - orange
	case 7: return cv::Scalar(0, 0, 255);   // truck - red
	default: return cv::Scalar(0, 255, 255);
	}
}

const char* getName(int classId) {
	switch (classId) {
	case 2: return "Car";
	case 5: return "Bus";
	case 7: return "Truck";
	default: return "Vehicle";
	}
}


int main(int argc, char** argv) {

	if (argc < 2) {
		std::cout << "\n";
		std::cout << "========================================\n";
		std::cout << "     YOLO Vehicle Detector\n";
		std::cout << "========================================\n";
		std::cout << "\nUsage: task1_detector <input>\n\n";
		std::cout << "Input:\n";
		std::cout << "  video.mp4    Video file\n";
		std::cout << "  camera       Default camera\n";
		std::cout << "  rtsp://...   RTSP stream\n";
		std::cout << "\nControls:\n";
		std::cout << "  ESC/Q - Quit\n";
		std::cout << "  S     - Screenshot\n";
		std::cout << "  R     - Toggle recording\n\n";
		return 0;
	}

	std::string source = argv[1];

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

	// Setup video writer for recording
	cv::VideoWriter writer;
	bool recording = true;
	std::string outputFile = "demo_detection.mp4";

	writer.open(outputFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
		videoFps, cv::Size(frameW, frameH));

	if (writer.isOpened())
		std::cout << "[OK] Recording: " << outputFile << "\n";
	else {
		std::cout << "[WARN] Cannot create output file\n";
		recording = false;
	}

	// Setup ONNX Runtime
	const ORTCHAR_T* modelPath = ORT_TSTR("models/yolo11n.onnx");
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "detector");
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


	std::string winName = "YOLO Vehicle Detector";
	cv::namedWindow(winName, cv::WINDOW_NORMAL);

	// Prepare input buffer
	std::vector<float> inputData(3 * INPUT_W * INPUT_H);

	double fps = 0.0 , avgFps = 0.0;
	int frameCount = 0 , screenshotNum = 0;
	std::cout << "[INFO] Press ESC to quit\n\n";

	cv::Mat frame;

	// Main loop
	while (cap.read(frame)) {
		auto startTime = std::chrono::high_resolution_clock::now();

		cv::Mat img = frame.clone();

		// Preprocess: resize, convert to RGB, normalize
		cv::Mat resized, rgb, blob;
		cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
		cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
		rgb.convertTo(blob, CV_32F, 1.0 / 255.0);

		// Convert to CHW format
		std::vector<cv::Mat> channels(3);
		cv::split(blob, channels);

		int chSize = INPUT_W * INPUT_H;
		for (int c = 0; c < 3; c++) {
			memcpy(inputData.data() + c * chSize, channels[c].data, chSize * sizeof(float));
		}

		// Create input tensor
		std::vector<int64_t> inputShape = { 1, 3, INPUT_H, INPUT_W };
		auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
			memInfo, inputData.data(), inputData.size(),
			inputShape.data(), inputShape.size()
		);

		// Run inference
		const char* inNames[] = { "images" };
		const char* outNames[] = { "output0" };

		auto output = session.Run(Ort::RunOptions{ nullptr },
			inNames, &inputTensor, 1,
			outNames, 1);

		float* data = output[0].GetTensorMutableData<float>();

		std::vector<cv::Rect> boxes;
		std::vector<float> scores;
		std::vector<int> classIds;

		float scaleX = (float)img.cols / INPUT_W;
		float scaleY = (float)img.rows / INPUT_H;

		for (int i = 0; i < NUM_PREDICTIONS; i++) {
			// Find best class score
			float bestScore = -1.0f;
			int bestClass = -1;

			for (int cid : VEHICLE_CLASSES) {
				float score = toProb(data[(4 + cid) * NUM_PREDICTIONS + i]);
				if (score > bestScore) {
					bestScore = score;
					bestClass = cid;
				}
			}

			if (bestScore < CONF_THRESHOLD) continue;

			float cx = data[0 * NUM_PREDICTIONS + i] * scaleX;
			float cy = data[1 * NUM_PREDICTIONS + i] * scaleY;
			float w = data[2 * NUM_PREDICTIONS + i] * scaleX;
			float h = data[3 * NUM_PREDICTIONS + i] * scaleY;

			int x = (int)(cx - w / 2);
			int y = (int)(cy - h / 2);
			int bw = (int)w;
			int bh = (int)h;

			x = std::max(0, x);
			y = std::max(0, y);
			bw = std::min(bw, img.cols - x);
			bh = std::min(bh, img.rows - y);

			if (bw <= 0 || bh <= 0) continue;

			boxes.emplace_back(x, y, bw, bh);
			scores.push_back(bestScore);
			classIds.push_back(bestClass);
		}

		// Apply NMS
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);

		for (int idx : indices) {
			cv::Rect box = boxes[idx];
			cv::Scalar color = getColor(classIds[idx]);

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

		cv::rectangle(img, cv::Point(10, 10), cv::Point(350, 90), cv::Scalar(0, 0, 0), cv::FILLED);

		cv::putText(img, cv::format("FPS: %.1f (Avg: %.1f)", fps, avgFps),
			cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

		cv::putText(img, cv::format("Detections: %d", (int)indices.size()),
			cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

		if (recording) {
			cv::putText(img, "REC", cv::Point(20, 85),
				cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
		}

		if (recording && writer.isOpened()) {
			writer.write(img);
		}

		cv::imshow(winName, img);
		frameCount++;

		int key = cv::waitKey(1) & 0xFF;

		if (key == 27 || key == 'q' || key == 'Q')
			break;

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

	// Cleanup
	cap.release();
	writer.release();
	cv::destroyAllWindows();

	std::cout << "\n[DONE]\n";
	std::cout << "  Frames: " << frameCount << "\n";
	std::cout << "  Avg FPS: " << (int)avgFps << "\n";
	std::cout << "  Output: " << outputFile << "\n";

	return 0;
}