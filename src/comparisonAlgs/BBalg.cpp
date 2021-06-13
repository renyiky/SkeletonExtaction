#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

#include "previousAlgs.hpp"

using namespace std;
using namespace cv;

#define REGISTER_NEIGHBORS(x, y) \
			int p1 = img.at<uchar>(x - 1, y - 1); \
			int p2 = img.at<uchar>(x - 1, y); \
			int p3 = img.at<uchar>(x - 1, y + 1); \
			int p4 = img.at<uchar>(x, y + 1); \
			int p = img.at<uchar>(x, y); \
			int p5 = img.at<uchar>(x + 1, y + 1); \
			int p6 = img.at<uchar>(x + 1, y); \
			int p7 = img.at<uchar>(x + 1, y - 1); \
			int p8 = img.at<uchar>(x, y - 1);

const uchar pixelBG = 0;
const uchar pixelFG = 1;
const uchar pixelContour = 2;
const uchar pixelSkeleton = 3;

bool loop = true;
int count = 0;

void preprocessing(Mat &img) {
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			REGISTER_NEIGHBORS(x, y)

			if (p == pixelFG && (p2 * p4 * p6 * p8 == pixelBG)) {
				img.at<uchar>(x, y) = pixelContour;

				// Noisy edge preparing
				if (p4 + p5 + p6 + p7 + p8 == 0 && p1 + p3 != 0 && p2 != 0) {
					img.at<uchar>(x, y) = pixelBG;
					img.at<uchar>(x - 1, y) = pixelContour;
				}
				else if (p1 + p2 + p6 + p7 + p8 == 0 && p3 + p5 != 0 && p4 != 0) {
					img.at<uchar>(x, y) = pixelBG;
					img.at<uchar>(x, y + 1) = pixelContour;
				}
				else if (p1 + p2 + p3 + p4 + p8 == 0 && p5 + p7 != 0 && p6 != 0) {
					img.at<uchar>(x, y) = pixelBG;
					img.at<uchar>(x + 1, y) = pixelContour;
				}
				else if (p2 + p3 + p4 + p5 + p6 == 0 && p1 + p7 != 0 && p8 != 0) {
					img.at<uchar>(x, y) = pixelBG;
					img.at<uchar>(x, y - 1) = pixelContour;
				}
			}
		}
	}
}

void peeling(Mat &img) {
	// Keeping strokes one-pixel width
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			if (img.at<uchar>(x, y) == pixelContour) {
				REGISTER_NEIGHBORS(x, y)

				if (p2 + p6 == 0 ||
					p4 + p8 == 0 ||
					(p1 != 0 && p2 + p8 == 0) ||
					(p3 != 0 && p2 + p4 == 0) ||
					(p5 != 0 && p4 + p6 == 0) ||
					(p7 != 0 && p6 + p8 == 0)) {
					img.at<uchar>(x, y) = pixelSkeleton;
				}
			}
		}
	}

	Mat newImg = img.clone();
	// Removing pixels of corners
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			if (img.at<uchar>(x, y) == pixelContour) {
				REGISTER_NEIGHBORS(x, y)

				if ((p1 + p2 + p8 == 0 && p4 == pixelContour && p6 == pixelContour && p5 == pixelFG) ||
					(p2 + p3 + p4 == 0 && p6 == pixelContour && p8 == pixelContour && p7 == pixelFG) ||
					(p4 + p5 + p6 == 0 && p2 == pixelContour && p8 == pixelContour && p1 == pixelFG) ||
					(p6 + p7 + p8 == 0 && p2 == pixelContour && p4 == pixelContour && p3 == pixelFG) ||
					((p2 == pixelFG && p6 == pixelBG) || (p6 == pixelFG && p2 == pixelBG)) ||
					((p8 == pixelFG && p4 == pixelBG) || (p4 == pixelFG && p8 == pixelBG))) {
					newImg.at<uchar>(x, y) = pixelBG;
					loop = true;
				}
			}
		}
	}

	img = newImg;

	// reset pixel condition
	for (int x = 0; x < img.rows; ++x) {
		for (int y = 0; y < img.cols; ++y) {
			if (img.at<uchar>(x, y) != pixelBG && img.at<uchar>(x, y) != pixelSkeleton)
				img.at<uchar>(x, y) = pixelFG;
		}
	}

}

void postprocessing(Mat &img) {
	// Two-pixel width solving
	bool flag = true;
	while (flag) {
		Mat newImg = img.clone();
		flag = false;
		for (int x = 1; x < img.rows - 1; ++x) {
			for (int y = 1; y < img.cols - 1; ++y) {
				if (img.at<uchar>(x, y) == pixelFG) {
					REGISTER_NEIGHBORS(x, y)

					if ((p1 + p2 + p8 == 0 && p5 == pixelFG) ||
						(p2 + p3 + p4 == 0 && p7 == pixelFG) ||
						(p4 + p5 + p6 == 0 && p1 == pixelFG) ||
						(p6 + p7 + p8 == 0 && p3 == pixelFG)) {
						flag = true;
						newImg.at<uchar>(x, y) = 0;
					}
				}
			}
		}
		img = newImg;
	}

	// Stairs solving
	for (int x = 1; x < img.rows - 1; ++x) {
		for (int y = 1; y < img.cols - 1; ++y) {
			if (img.at<uchar>(x, y) == pixelFG) {
				REGISTER_NEIGHBORS(x, y)

				if ((p2 * p8 != 0 && p1 == pixelBG) ||
					(p2 * p4 != 0 && p3 == pixelBG) ||
					(p4 * p6 != pixelBG && p5 == pixelBG) ||
					(p6 * p8 != pixelBG && p7 == pixelBG)) {
					img.at<uchar>(x, y) = pixelBG;
				}
			}
		}
	}
}

void binarize(Mat &img) {
	for (int x = 0; x < img.rows; ++x) {
		for (int y = 0; y < img.cols; ++y) {
			if(img.at<uchar>(x, y) != 0) img.at<uchar>(x, y) = 1;
		}
	}
}

Mat BBalg(Mat img) {
	binarize(img);

	while (loop) {
		loop = false;
		preprocessing(img);
		peeling(img);
	}
	postprocessing(img);
	
	Mat ret = Mat::zeros(img.rows, img.cols, CV_8U);
	for (int x = 0; x < img.rows; ++x) {
		for (int y = 0; y < img.cols; ++y) {
			if (img.at<uchar>(x, y) == pixelFG || img.at<uchar>(x, y) == pixelSkeleton)
				ret.at<uchar>(x, y) = 255;
		}
	}
	return ret;
}