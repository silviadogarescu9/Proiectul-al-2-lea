// g++ cod.cpp -std=c++11  -o cod `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;
 
int main(int argc,char** argv) {

	// creeaza o noua instanta de FacemarkLBF
	Ptr<Facemark> facemark = createFacemarkLBF();

	//imgainea input
	Mat img_orig = imread("jk1.jpg", IMREAD_COLOR);

	//imaginea originala cu filtrul alb_negru
	Mat image = imread("jk1.jpg", IMREAD_GRAYSCALE);

	Mat output_sepia = imread("jk1.jpg", IMREAD_COLOR);

	//variabile pentrul filtru cartoon
	Mat img_rgb = imread("jk1.jpg", IMREAD_COLOR);
	Mat img_color = img_rgb;
	Mat img_bilateral_dest;
	Mat img_gray;
	Mat img_blur;
	Mat img_edge;
	Mat img_edge_conv;
	Mat img_cartoon = img_rgb;
	
	//filtru sepia
	cv::Mat kernel =
		(cv::Mat_<float>(3, 3)
			<<
			0.272, 0.534, 0.131,
			0.349, 0.686, 0.168,
			0.393, 0.769, 0.189);

	cv::transform(img_orig, output_sepia, kernel);

	std::vector<cv::Rect> faces;

	//detecteaza landmarks
	std::vector<std::vector<Point2f> > landmarks;

	//loadeaza trained model
	facemark->loadModel("lbfmodel.yaml");

	CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	faceDetector.detectMultiScale(image, faces);
	facemark->fit(image, faces, landmarks);

	/* printeaza ce e in landmark
	for(int j=0;j<landmarks[0].size();j++) {
			cout<< landmarks[0][j];
	}*/

	//rezultatatul desenarii pe imagine a facemark-urilor
	for(int j=0;j<faces.size();j++){
		face::drawFacemarks(img_orig, landmarks[j], Scalar(0,0,255));
	}
	imshow("facemark", img_orig);
	imshow("filtru_alb_negru", image);
	imshow("filtru_sepia", output_sepia);
	
	//Cod filtru cartoon

	//edge-aware smoothing cu filtru bilateral
	for(int i = 0; i < 2; i++) {
		pyrDown(img_color, img_color, Size((img_color.cols+1)/2, (img_color.rows+1)/2), BORDER_DEFAULT);
	}
	
	for(int i = 0; i < 7; i++) {
		bilateralFilter(img_color, img_bilateral_dest, 9,  9, 7);
	}
	img_color = img_bilateral_dest;
	
	for(int i = 0; i < 2; i++) {
		pyrUp(img_color, img_color, Size(img_color.cols*2, (img_color.rows*2)), BORDER_DEFAULT);
	}

	//reduce sunetul cu filtru median	
	cvtColor(img_rgb, img_gray,COLOR_RGB2GRAY, 0 );
	medianBlur(img_gray,img_blur,7);

	//creaza un edge mask folosind adaptive thresholding	
	adaptiveThreshold(img_blur, img_edge, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 2);


	//combina imaginea colorata cu edge mask-ul
	cvtColor(img_edge, img_edge_conv, COLOR_GRAY2RGB, 0);
	bitwise_and(img_rgb, img_edge_conv, img_cartoon, noArray());

	imshow("filtru_cartoon", img_cartoon);
	waitKey();
	
    return 0;
}
