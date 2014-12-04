// DRFI.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include "SalDRFI.h"
//#include "CmTimer.h"
//#include "CmEvaluation.h"
// TestDRFI.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
//#include "./CmLib/CmTimer.h"
//#include "./CmLib/CmFile.h"
//#include "./CmLib/CmEvaluation.h"

void processImg(CStr &imgPath);
void processDataset(int argc, char* argv[]);

// C:/WkDir/Saliency/FT1000/Imgs/*.jpg  DrfiModel.data  C:/WkDir/Saliency/FT1000/SalMaps/  C:/WkDir/Saliency/FT1000/Imgs/*.png
int main(int argc, char* argv[])
{
	processImg("0_0_272.png");
	// processDataset(argc, argv);
	return 0;
}

void processImg(CStr &imgPath)
{ 
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	SalDRFI drfi;
	drfi.load( "drfiModelCpp.data" );		// http://jianghz.com/drfi/drfiModelCpp.data

	Mat img3u = imread(imgPath);
	Mat sal1f;
	// for( int ix = 0; ix < 20; ++ix ) // check if openmp works
		sal1f = drfi.getSalMap(img3u);
	imshow("Image", img3u);
	imshow("Saliency", sal1f);
	waitKey(0);

	return;
	//_CrtDumpMemoryLeaks();
}

//void processDataset(int argc, char* argv[])
//{	
//	if (argc < 4 || argc > 6){
//		printf("Usage: %s ImageNames ModelName OutDir [GtDir]\n", _S(CmFile::GetName(argv[0])));
//		printf("\tSample: %s ./Imgs/*.jpg DrfiModel.data ./SalMaps/ ./Imgs/*.png\n", _S(CmFile::GetName(argv[0])));
//		printf("\tImageNames: input image names\n");
//		printf("\tModelName: input model file name. Could be download from: http://mmcheng.net/mftp/Data/DrfiModel.data \n");
//		printf("\tOutDir: output saliency maps directory\n");
//		printf("\tGtDir: an optional parameter. If exists, it could be the ground truth files saved in png for evaluation\n");
//		return;
//	}
//	CStr imgW = argv[1], modelName = argv[2], outDir = argv[3], gtW = argc == 5 ? argv[4] : "";
//	if (!CmFile::FileExist(modelName)){
//		printf("Model file not exist. Please download it from: http://mmcheng.net/mftp/Data/DrfiModel.data \n");
//		return;
//	}
//
//	// SalDRFI::MatlabDat2DataFile("Model.mat", modelName); // Convert Matlab file to C++ readable file
//	SalDRFI drfi;
//	drfi.load(modelName);  // http://mmcheng.net/mftp/Data/DrfiModel.data
//
//	std::string extention = imgW.substr( imgW.find_last_of( "." ) );
//	CmFile::MkDir(outDir);
//	vecS namesNE, des;
//	string imgDir;
//	int imgNum = CmFile::GetNamesNE(imgW, namesNE, imgDir);
//	printf("%d images found in %s\n", imgNum, _S(imgDir + "*" + extention));
//
//
//	CmTimer tm("Get saliency maps time");
//	tm.Start();
//
//	const double sigma = 0.9;
//	const double k = 200;
//	const double minSize = 200;
//
//	int thrNum = omp_get_max_threads();
////	if (imgNum > thrNum){
////#pragma omp parallel for
////		for (int i = 0; i < imgNum; i++)
////			imwrite(outDir + namesNE[i] + "_DRFI.png", drfi.getSalMap(imread(imgDir + namesNE[i] + extention))*255);
////			// imwrite(outDir + namesNE[i] + "_DRFI.png", drfi.getSalMap(imread(imgDir + namesNE[i] + extention), sigma, k, minSize)*255 );
////	}
////	else
//		for (int i = 0; i < imgNum; i++)
//			imwrite(outDir + namesNE[i] + "_DRFI.png", drfi.getSalMap(imread(imgDir + namesNE[i] + extention))*255);
//			// imwrite(outDir + namesNE[i] + "_DRFI.png", drfi.getSalMap(imread(imgDir + namesNE[i] + extention), sigma, k, minSize)*255 );
//
//	tm.Stop();
//	printf("Speed: %g seconds = %g fps\n", tm.TimeInSeconds()/imgNum, imgNum/tm.TimeInSeconds());
//
//	if (gtW.size()){
//		CStr resName = CmFile::GetFatherFolder(imgDir) + "EvaluationResults.m";
//		printf("Evaluation statistics shown in: %s\n", _S(resName));
//		CmEvaluation::Evaluate(gtW, outDir, resName, "DRFI");
//	}
//}
