// g++ cod.cpp -std=c++11  -o cod `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/gapi/own/types.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace cv::face;
 
int main(int argc,char** argv) {

	if(argc < 2) {
		cout << "Introdu numele selfie-ului" << endl;
		return 2;
	}

	//pentru randomizarea caricaturii
	int rrand_os, crand_os, rrand_od, crand_od,rrand_bar;
	int rrand_g, crand_g,rrand_nas, plusminus;

	// creeaza o noua instanta de FacemarkLBF
	Ptr<Facemark> facemark = createFacemarkLBF();
	//imagine pentru facemark
	Mat image = imread(argv[1], IMREAD_GRAYSCALE);

	//imgainea input
	Mat img_orig = imread(argv[1], IMREAD_COLOR);
	Mat output_alb_negru = imread(argv[1], IMREAD_GRAYSCALE);
	Mat output_sepia = imread(argv[1], IMREAD_COLOR);

	//variabile pentrul filtrul cartoon
	Mat img_rgb = imread(argv[1], IMREAD_COLOR);
	Mat img_color = img_rgb;
	Mat img_bilateral_dest;
	Mat img_gray;
	Mat img_blur;
	Mat img_edge;
	Mat img_edge_conv;
	Mat img_cartoon = img_rgb;
	
	//necesar pentru filtrul sepia
	cv::Mat kernel =
		(cv::Mat_<float>(3, 3)
			<<
			0.272, 0.534, 0.131,
			0.349, 0.686, 0.168,
			0.393, 0.769, 0.189);

	//pentru caricatura
	Mat original = imread(argv[1], IMREAD_COLOR);
	Mat modificat = imread(argv[1], IMREAD_COLOR);

/*facemark*/
	std::vector<cv::Rect> faces;

	//detecteaza landmarks
	std::vector<std::vector<Point2f> > landmarks;

	//loadeaza trained model
	facemark->loadModel("lbfmodel.yaml");

	CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	faceDetector.detectMultiScale(image, faces);
	facemark->fit(image, faces, landmarks);

	//desenare facemark-uri
	for(int j=0;j<faces.size();j++){
		face::drawFacemarks(img_orig, landmarks[j], Scalar(0,0,255));
	}
	
/* caricatura */
	
	//partea de randoizare a caricaturii
	srand (time(NULL));
	rrand_os = rand() %  5 + 2;
	crand_os = rand() %  5 + 2;
	rrand_od = rand() %  3 + 2;
	crand_od = rand() %  7 + 2;
	rrand_bar = rand() %  3 + 4;
	rrand_g = rand() %  3 + 4;
	crand_g = rand() %  3 + 4;
	rrand_nas = rand() %  3 + 8;
	plusminus = rand() % 2;

	if(plusminus == 0) {
		crand_os = crand_os * (-1);
		rrand_od = rrand_od * (-1);
		crand_od = crand_od * (-1);
	}

	plusminus = rand() % 2;

	if(plusminus == 0) {
		rrand_bar = rrand_bar * (-1);
		crand_g = crand_g * (-1);
		rrand_g = rrand_g * (-1);
		rrand_nas = rrand_nas * (-1);
	}

	for(int r = 0; r < original.rows; r++){
		for(int c = 0; c < original.cols; c++) {
		
			//ochiul stang
			if((r > int(landmarks[0][37].y) -5 && r < int(landmarks[0][40].y )+ 12)   &&
				( c > int(landmarks[0][36].x ) -8  && c < int(landmarks[0][39].x ) +4  )){

				modificat.at<Vec3b>(r,c)[0] = original.at<Vec3b>(r + rrand_os , c + crand_os)[0]; // blue channel
				modificat.at<Vec3b>(r,c)[1] = original.at<Vec3b>(r + rrand_os , c + crand_os)[1]; // green channel
				modificat.at<Vec3b>(r,c)[2] = original.at<Vec3b>(r + rrand_os , c + crand_os)[2]; // red channel
				continue;

			}
			//ochiul drept
			if((r > int(landmarks[0][43].y) -5 && r < int(landmarks[0][46].y )+ 12)   &&
				( c > int(landmarks[0][42].x ) -8  && c < int(landmarks[0][45].x ) +4  )) {

				modificat.at<Vec3b>(r,c)[0] = original.at<Vec3b>(r + rrand_od, c + crand_od)[0];
				modificat.at<Vec3b>(r,c)[1] = original.at<Vec3b>(r + rrand_od, c + crand_od)[1];
				modificat.at<Vec3b>(r,c)[2] = original.at<Vec3b>(r + rrand_od, c + crand_od)[2];
				continue;

			}

			//barbie
			if((r > int(landmarks[0][5].y) && r < int(landmarks[0][8].y ))   &&
				( c > int(landmarks[0][5].x - 10) && c < int(landmarks[0][11].x + 10) )) {

				modificat.at<Vec3b>(r,c)[0] = original.at<Vec3b>(r + rrand_bar, c)[0];
				modificat.at<Vec3b>(r,c)[1] = original.at<Vec3b>(r + rrand_bar, c)[1];
				modificat.at<Vec3b>(r,c)[2] = original.at<Vec3b>(r + rrand_bar, c)[2];
				continue;

			}

			//gura
			if((r > int(landmarks[0][50].y) && r < int(landmarks[0][57].y ))   &&
				( c > int(landmarks[0][48].x) && c < int(landmarks[0][54].x ) )) {

				modificat.at<Vec3b>(r,c)[0] = original.at<Vec3b>(r + rrand_g, c + crand_g)[0];
				modificat.at<Vec3b>(r,c)[1] = original.at<Vec3b>(r + rrand_g, c + crand_g)[1];
				modificat.at<Vec3b>(r,c)[2] = original.at<Vec3b>(r + rrand_g, c + crand_g)[2];
				continue;

			}

			//nas
			if((r > int(landmarks[0][29].y) && r < int(landmarks[0][33].y ) + 5)   &&
				( c > int(landmarks[0][31].x ) - 11 && c < int(landmarks[0][35].x ) + 11 )) {

				modificat.at<Vec3b>(r,c)[0] = original.at<Vec3b>(r + rrand_nas, c)[0];
				modificat.at<Vec3b>(r,c)[1] = original.at<Vec3b>(r + rrand_nas, c)[1];
				modificat.at<Vec3b>(r,c)[2] = original.at<Vec3b>(r + rrand_nas, c)[2];
				continue;

			}

			modificat.at<Vec3b>(r,c)[0] = original.at<Vec3b>(r,c)[0];
			modificat.at<Vec3b>(r,c)[1] = original.at<Vec3b>(r,c)[1];
			modificat.at<Vec3b>(r,c)[2] = original.at<Vec3b>(r,c)[2];
		}
	}


/* filtru cartoon */
	img_rgb = modificat;
	img_color = img_rgb;

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

/*aplicarea filtrului sepie pe caricatura */
	transform(modificat, output_sepia, kernel);
	
/*aplicarea filtrului alb-negru pe caricatura */
	cvtColor(modificat, output_alb_negru,COLOR_RGB2GRAY, 0);

	imshow("originalul", original);
	imshow("facemark", img_orig);
	imshow("caricatura", modificat);
	imshow("filtru_sepia_pe_caricatura", output_sepia);
	imshow("filtru_alb_negru", output_alb_negru);
	imshow("filtru_cartoon", img_cartoon);
	
	imwrite("caricatura.jpg", modificat, std::vector< int >() );
	imwrite("caricatura_sepia.jpg", output_sepia, std::vector< int >() );
	imwrite("caricatura_alb_negru.jpg", output_alb_negru, std::vector< int >() );
	imwrite("caricatura_cartoon.jpg", img_cartoon, std::vector< int >() );

	waitKey();
	
    return 0;
}
