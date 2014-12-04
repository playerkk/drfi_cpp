//#include "stdafx.h"
#include "SalDRFI.h"
#include "reg_RF.h"

Mat SalDRFI::getSalMap(CMat &_img3u)
{
	Mat img3f, imgBlur3f, img3u;
	_img3u.convertTo(img3f, CV_32F, 1.0/255);
	GaussianBlur(_img3u, img3u, Size(5, 5), 0);
	img3u.convertTo(imgBlur3f, CV_32F, 1.0/255);

	CV_Assert_(_dataLoaded, ("Model data needs to be loaded before getting saliency map\n"));
	vecM smaps1d(_N);
	int NUM_P = img3u.rows * img3u.cols;
	RegionFeature regF; // Get image data
	regF.setImg(img3u, imgBlur3f);

	//CmTimer tm("Tm"); tm.Start();

	// Get saliency maps for each scale
#pragma omp parallel for
	for (int i = 0; i < _N; i++){
		_w[i] = 1.0;
		Mat segIdx1i;
		double segSigma = _segPara1d.at<double>(0, i), segK = _segPara1d.at<double>(1, i), segMin = _segPara1d.at<double>(2, i);
		int numReg = SegmentImage(img3f, segIdx1i, segSigma, segK/255, cvRound(segMin));
		Mat spSalData1d = regF.getFeature(segIdx1i, numReg);
		
		vecD regSal1d = regRandomForestPredict(spSalData1d);// * _w[i];
		for( int ii = 0; ii < regSal1d.size(); ++ii )
			regSal1d[ii] *= _w[i];

		smaps1d[i].create(img3u.size(), CV_64F);
		double *sal = (double*)smaps1d[i].data;
		const int *regIdx = (int*)segIdx1i.data;
		for (int p = 0; p < NUM_P; p++)
			sal[p] = regSal1d[regIdx[p]];
	}

	// Merge saliency maps at each scale
	Mat smap1d = smaps1d[0];
	for (int i = 1; i < _N; i++){
		double *totalM = (double*)smap1d.data, *subM = (double*)smaps1d[i].data;
#pragma omp parallel for
		for (int j = 0; j < NUM_P; j++)
			totalM[j] += subM[j];
	}
	normalize(smap1d, smap1d, 0, 1, CV_MINMAX);
	//tm.StopAndReport();
	return smap1d;
}

cv::Mat SalDRFI::getSalMap( CMat &_img3u, double segSigma, double segK, double segMin )
{
	Mat img3f, imgBlur3f, img3u;
	_img3u.convertTo(img3f, CV_32F, 1.0/255);
	GaussianBlur(_img3u, img3u, Size(5, 5), 0);
	img3u.convertTo(imgBlur3f, CV_32F, 1.0/255);

	CV_Assert_(_dataLoaded, ("Model data needs to be loaded before getting saliency map\n"));
	vecM smaps1d(_N);
	int NUM_P = img3u.rows * img3u.cols;
	RegionFeature regF; // Get image data
	regF.setImg(img3u, imgBlur3f);

	//CmTimer tm("Tm"); tm.Start();

	Mat segIdx1i;
	int numReg = SegmentImage(img3f, segIdx1i, segSigma, segK/255.0, cvRound(segMin));
	Mat spSalData1d = regF.getFeature(segIdx1i, numReg);

	//{
	//	cv::Mat tempSmap;
	//	tempSmap.create(segIdx1i.rows, segIdx1i.cols, CV_64F);
	//	for( size_t c = 0; c < segIdx1i.cols; ++c )
	//	{
	//		for( size_t r = 0; r < segIdx1i.rows; ++r )
	//		{
	//			int regInd = segIdx1i.at<int>( r, c );
	//			tempSmap.at<double>(r, c) = spSalData1d.at<double>( regInd, 3 );
	//		}
	//	}

	//	normalize(tempSmap, tempSmap, 0, 1, CV_MINMAX);

	//	cv::imshow( "visualize", tempSmap );
	//	cv::waitKey(0);
	//	//std::ofstream out( "feat.bin", std::ios::binary );
	//	//out.write( (char*)(&spSalData1d.rows), sizeof(int) );
	//	//out.write( (char*)(&spSalData1d.cols), sizeof(int) );

	//	//for( int c = 0; c < spSalData1d.cols; ++c )
	//	//{
	//	//	for( int r = 0; r < spSalData1d.rows; ++r )
	//	//	{
	//	//		out.write( (char*)(&spSalData1d.at<double>(r, c)), sizeof(double) );			
	//	//	}
	//	//}
	//	//out.close();
	//}

	vecD regSal1d = regRandomForestPredict(spSalData1d);// * _w[i];

	Mat smap1d;
	smap1d.create(img3u.size(), CV_64F);
	double *sal = (double*)smap1d.data;
	const int *regIdx = (int*)segIdx1i.data;
	for (int p = 0; p < NUM_P; p++)
		sal[p] = regSal1d[regIdx[p]];

	normalize(smap1d, smap1d, 0, 1, CV_MINMAX);
	//tm.StopAndReport();
	return smap1d;
}

vecD SalDRFI::regRandomForestPredict(CMat &spSalData)
{
	//  n_size                   p_size 
	int feaNum = spSalData.rows, feaDim = spSalData.cols;
	vecD regSal(feaNum);
	Mat cat1i = Mat::ones(1, feaDim, CV_32S), nodex1i(1, feaNum, CV_32S);
	int maxCat = 1, keepPred = 0, doProx = 0, nodes = 0;
	double allPred = 0, proxMat = 0;
	regForest((double*)spSalData.data, &regSal[0], &feaDim, &feaNum, &_NumT, (int*)_lDau1i.data,
		(int*)_rDau1i.data, (char*)_nodeStatus1c.data, &_NumN, (double*)_upper1d.data,
		(double*)_avNode1d.data, (int*)_mBest1i.data, &_ndTree[0], (int*)cat1i.data,
		maxCat, &keepPred, &allPred, doProx, &proxMat, &nodes, (int*)nodex1i.data);
	return regSal;
}

bool matRead(FILE *f, Mat& M)
{
	if (f == NULL)
		return false;
	char buf[8];
	int pre = (int)fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file: %d:%s\n", __LINE__, __FILE__);
		return false;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	M = Mat(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	return true;
}

bool matRead( const string& filename, Mat& M){
	FILE* f = fopen(_S(filename), "rb");
	bool res = matRead(f, M);
	if (f != NULL)
		fclose(f);
	return res;
}

void SalDRFI::load(CStr dataFile)
{
	FILE *f = fopen(_S(dataFile), "rb");
	CV_Assert_(f != NULL, ("Can't open file %s\n", _S(dataFile)));
	char buf[100];
	int pre = (int)fread(buf, sizeof(char), SZ_MARKER, f);
	if (strncmp(buf, MARKER, SZ_MARKER) != 0)	{
		printf("Invalidate DrfiData file at %d:%s\n", __LINE__, __FILE__);
		return;
	}
	const int numSzData = 3;
	int szData[3];
	fread(szData, sizeof(int), numSzData, f);
	_N = szData[0], _NumN = szData[1], _NumT = szData[2];
	Mat w, ndTree;
	matRead(f, w);
	matRead(f, _segPara1d);
	matRead(f, _lDau1i);
	matRead(f, _rDau1i);
	matRead(f, _mBest1i);
	matRead(f, _nodeStatus1c);
	matRead(f, _upper1d);
	matRead(f, _avNode1d);
	matRead(f, _mlFilters15d);
	matRead(f, ndTree);
	_w = w;
	_ndTree = ndTree;
	fclose(f);	
	_dataLoaded = true;

	RegionFeature::setFilter(_mlFilters15d);
}

// Write matrix to binary file
bool matWrite(FILE *f, CMat& _M)
{
	Mat M;
	_M.copyTo(M);
	if (f == NULL || M.empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, f);
	int headData[3] = {M.cols, M.rows, M.type()};
	fwrite(headData, sizeof(int), 3, f);
	fwrite(M.data, sizeof(char), M.step * M.rows, f);
	return true;
}

bool matWrite(CStr& filename, CMat& M){
	FILE* f = fopen(_S(filename), "wb");
	bool res = matWrite(f, M);
	if (f != NULL)
		fclose(f);	
	return res;
}

void SalDRFI::save(CStr dataFile)
{
	FILE *f = fopen(_S(dataFile), "wb");
	CV_Assert_(f != NULL, ("Can't open file %s\n", _S(dataFile)));
	fwrite(MARKER, sizeof(char), SZ_MARKER, f);
	int szData[] = {_N, _NumN, _NumT};
	fwrite(szData, sizeof(int), sizeof(szData)/sizeof(int), f);
	matWrite(f, Mat(_w));
	matWrite(f, _segPara1d);
	matWrite(f, _lDau1i);
	matWrite(f, _rDau1i);
	matWrite(f, _mBest1i);
	matWrite(f, _nodeStatus1c);
	matWrite(f, _upper1d);
	matWrite(f, _avNode1d);
	matWrite(f, _mlFilters15d);
	matWrite(f, Mat(_ndTree));
	fclose(f);
}

// #define _USE_MATLAB

void SalDRFI::MatlabDat2DataFile(CStr matlabData, CStr dataFile)
{
#ifdef _USE_MATLAB
	CmMatFile matFile(matlabData);
	SalDRFI model;
	model._N = matFile.get<int>("N");
	model._NumN = matFile.get<int>("NumN");
	model._NumT = matFile.get<int>("NumT");

	model._w = matFile.getMat("w", CV_64FC1, false); // Not calling transpose to be compilable with the original code.
	model._segPara1d = matFile.getMat("segPara", CV_64FC1, false);

	model._lDau1i = matFile.getMat("lDau", CV_32S, false);
	model._rDau1i = matFile.getMat("rDau", CV_32S, false);
	model._mBest1i = matFile.getMat("mBest", CV_32S, false);

	Mat buggedMat = matFile.getMat("nodeStatus", CV_USRTYPE1, false); // To be compatible with the bug in the old code.
	int numElements = buggedMat.cols * buggedMat.rows;
	model._nodeStatus1c = Mat(buggedMat.size(), CV_8S);
	CV_Assert(model._nodeStatus1c.isContinuous());
	memcpy(model._nodeStatus1c.data, buggedMat.data, numElements);
	uchar* laterHalf = buggedMat.data + numElements;
	for (int i = 0; i < numElements; i++)
		CV_Assert(laterHalf[i] == 0);

	model._upper1d = matFile.getMat("upper", CV_64F, false);
	model._avNode1d = matFile.getMat("avNode", CV_64F, false);

	model._mlFilters15d = matFile.getMat("mlFilters", CV_64FC(15), true);
	model._ndTree = matFile.getMat("ndTree", CV_32S, false);

	model.save(dataFile);
#endif 
}