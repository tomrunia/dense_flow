#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::gpu;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y,
							   Mat &img_x, Mat &img_y,
							   double lowerBound, double higherBound)
{

#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
#undef CAST

}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
	{
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f &fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
				 color);
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
	}
}


static cv::Mat makeColorWheel()
{
	// Read more about the color wheel on this page:
	// http://members.shaw.ca/quadibloc/other/colint.htm

	const int RY = 15;
	const int YG = 6;
	const int GC = 4;
	const int CB = 11;
	const int BM = 13;
	const int MR = 6;

	int ncols = RY + YG + GC + CB + BM + MR;

	// Initialize the color wheel as a matrix
	Mat color_wheel = Mat::zeros(ncols, 3, CV_8U);

	int col = 0;

	// Red to Yellow
	for (int col = 0; col < RY; col++)
	{
		color_wheel.at<uchar>(col, 0) = 255;
		color_wheel.at<uchar>(col, 1) = uchar(floor(255*col/float(RY)));
	}

	// Yellow to Green
	for (int col = RY; col < RY+YG; col++)
	{
		int index = col - RY;
		color_wheel.at<uchar>(col, 0) = uchar(255 - floor((255*index)/float(YG)));
		color_wheel.at<uchar>(col, 1) = 255;
	}

	// Green to Cyan
	for (int col = RY+YG; col < RY+YG+GC; col++)
	{
		int index = col - (RY+YG);
		color_wheel.at<uchar>(col, 1) = 255;
		color_wheel.at<uchar>(col, 2) = uchar(floor((255*index)/float(GC)));
	}

	// Cyan to Blue
	for (int col = RY+YG+GC; col < RY+YG+GC+CB; col++)
	{
		int index = col - (RY+YG+GC);
		color_wheel.at<uchar>(col, 1) = uchar(255 - floor((255*index)/float(CB)));
		color_wheel.at<uchar>(col, 2) = 255;
	}

	// Blue to Magenta
	for (int col = RY+YG+GC+CB; col < RY+YG+GC+CB+BM; col++)
	{
		int index = col - (RY+YG+GC+CB);
		color_wheel.at<uchar>(col, 0) = uchar(floor((255*index)/float(BM)));
		color_wheel.at<uchar>(col, 2) = 255;

	}

	// Magenda to Red
	for (int col = RY+YG+GC+CB+BM; col < ncols; col++)
	{
		int index = col - (RY+YG+GC+CB+BM);
		color_wheel.at<uchar>(col, 0) = 255;
		color_wheel.at<uchar>(col, 2) = uchar(255 - floor((255*index)/float(MR)));
	}

	return color_wheel;
}

static cv::Mat convertToColorImage(const cv::Mat& flowX, const cv::Mat& flowY, const cv::Mat& color_wheel)
{

	const int cw_nrows = color_wheel.rows;
	const int cw_ncols = color_wheel.cols;

	// Output image
	cv::Mat color_image = cv::Mat(flowX.size(), CV_8UC3);

	for(int y = 0; y < flowX.rows; y++)
	{
		for (int x = 0; x < flowX.cols; x++)
		{

			const double u = flowX.at<float>(y, x);  // flow_x
			const double v = flowY.at<float>(y, x);  // flow_y

			const double radius = sqrt(pow(u,2)+pow(v,2));
			const double angle = atan2(-v, -u) / M_PI;

			// Mapping from atan2 = [-1, +1] to [0, cv_nrows]
			const double fk = (angle+1)/2.0 * (cw_nrows-1);
			const int k0 = floor(fk);
			const int k1 = k0+1;

			const double diff = fk-k0;

			// Iterate over color dimensions (RGB, 3)
			for (int i = 0 ; i < cw_ncols; i++)
			{
				const double color0 = double(color_wheel.at<uchar>(k0, i)) / 255.0;
				const double color1 = double(color_wheel.at<uchar>(k1, i)) / 255.0;

				// Interpolation
				double color = (1-diff)*color0 + (diff*color1);

				if (radius < 1.0)
				{
					// Increase saturation with the radius
					color = 1.0-radius*(1.0-color);
					//std::cout << "RADIUS < 1.0" << std::endl;
				}
				else
				{
					// Radius out of range, decrease it.
					//std::cout << "DECREASE" << std::endl;
					color *= 0.75;
				}

				// RGB to BGR
				int channel = 1;
				channel = (i == 0) ? 2 : channel;
				channel = (i == 2) ? 0 : channel;

				// Set pixel in the color image
				color_image.at<cv::Vec3b>(y, x)[channel] = uchar(floor(255.0*color));

			}

			//std::cout << color_image.at<cv::Vec3b>(y, x) << std::endl;
		}
	}

	return color_image;
}


int main(int argc, char** argv){
	// IO operation
	const char* keys =
			{
					"{ f  | vidFile      | ex2.avi | filename of video }"
					"{ x  | xFlowFile    | flow_x | filename of flow x component }"
					"{ y  | yFlowFile    | flow_y | filename of flow x component }"
					"{ i  | imgFile      | flow_i | filename of flow image}"
					"{ o  | rgbOutput    | rgb_dir | output dir of RGB flow image}"
					"{ b  | bound | 15 | specify the maximum of optical flow}"
					"{ t  | type | 0 | specify the optical flow algorithm }"
					"{ d  | device_id    | 0  | set gpu id}"
					"{ s  | step  | 1 | specify the step for frame sampling}"
					"{ c  | colorMap | 1 | apply color map for nice visualization}"
			};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	string rgbOutputDir = cmd.get<string>("rgbOutput");
	bool applyColorMap = cmd.get<int>("colorMap");

	int bound = cmd.get<int>("bound");
	int type  = cmd.get<int>("type");
	int device_id = cmd.get<int>("device_id");
	int step = cmd.get<int>("step");


	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
	GpuMat frame_0, frame_1, flow_u, flow_v;

	setDevice(device_id);
	FarnebackOpticalFlow alg_farn;
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

	// Custom coloring using color wheel
	cv::Mat color_wheel = makeColorWheel();

	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			frame_num++;

			int step_t = step;
			while (step_t > 1){
				capture >> frame;
				step_t--;
			}
			continue;
		}

		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		//  Mat prev_grey_, grey_;
		//  resize(prev_grey, prev_grey_, Size(453, 342));
		//  resize(grey, grey_, Size(453, 342));
		frame_0.upload(prev_grey);
		frame_1.upload(grey);


		// GPU optical flow
		switch(type){
			case 0:
				alg_farn(frame_0,frame_1,flow_u,flow_v);
				break;
			case 1:
				alg_tvl1(frame_0,frame_1,flow_u,flow_v);
				break;
			case 2:
				GpuMat d_frame0f, d_frame1f;
				frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
				frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
				alg_brox(d_frame0f, d_frame1f, flow_u,flow_v);
				break;
		}

		flow_u.download(flow_x);
		flow_v.download(flow_y);

		// Output optical flow
		Mat imgX(flow_x.size(),CV_8UC1);
		Mat imgY(flow_y.size(),CV_8UC1);
		convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"%05d.jpg",int(frame_num));

		// Output image
		cv::Mat bgr_image = cv::Mat::zeros(imgX.size(), CV_8UC3);


		if (applyColorMap)
		{

			// Initialize matrices
			cv::Mat imgX_f, imgX_sq = cv::Mat::zeros(imgX.size(), CV_32FC1);
			cv::Mat imgY_f, imgY_sq = cv::Mat::zeros(imgY.size(), CV_32FC1);
			imgX.convertTo(imgX_f, CV_32FC1);
			imgY.convertTo(imgY_f, CV_32FC1);

			// Compute u^2 and v^2
			cv::pow(imgX_f, 2.0, imgX_sq);
			cv::pow(imgY_f, 2.0, imgY_sq);

			// Compute magnitude: sqrt(u^2 + v^2)
			cv::Mat radius = cv::Mat::zeros(imgX.size(), CV_32FC1);
			cv::sqrt(imgX_sq + imgY_sq, radius);

			// Find the maximal radius
			double min_radius, max_radius;
			cv::minMaxLoc(radius, &min_radius, &max_radius);

			// Normalize by maximum radius
			imgX_f /= max_radius;
			imgY_f /= max_radius;

			// Compute optical flow image with color wheel
			bgr_image = convertToColorImage(imgX_f, imgY_f, color_wheel);

		}
		else
		{

			// Simply write flowX and flowY to the first two channels of the image
			// We leave the last channel empty (i.e. zeros)
			std::vector<Mat> flow_images(3);
			flow_images.at(0) = imgX;
			flow_images.at(1) = imgY;
			flow_images.at(2) = Mat::zeros(imgX.size(), CV_8UC1);
			cv::merge(flow_images, bgr_image);
		}

		// And finally write the output image to disk
		cv::imwrite(rgbOutputDir + std::string(tmp), bgr_image);

		// Optionally for visualizing the flow
		//cv::imshow("flow", bgr_image);
		//cv::waitKey(20);

		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1){
			capture >> frame;
			step_t--;
		}
	}
	return 0;
}