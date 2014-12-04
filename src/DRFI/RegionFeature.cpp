//#include "stdafx.h"
#include "CmDefinition.h"
#include "RegionFeature.h"

//#define USE_HUAIZU_ORDER

vecM RegionFeature::mlFilters1d;
const RegionFeature::S_QUANTIZE_FUNC RegionFeature::sqFuns[] = {SQ_BGR, SQ_HSV, SQ_Lab};

void RegionFeature::setFilter(CMat &mlFilters15d)
{
	split(mlFilters15d, mlFilters1d);
	CV_Assert(DIM_TEX == mlFilters1d.size());
}

void cvtColorBgr3f2Gray1u1d(CMat &img3u, Mat &gray1d, Mat &gray1u, double scale = 1.0/255)
{
	CV_Assert(img3u.isContinuous());
	int N = img3u.cols * img3u.rows;
	const Vec3b* img = (Vec3b*)img3u.data;
	gray1d.create(img3u.size(), CV_64F), gray1u.create(img3u.size(), CV_8U);
	double *grayPd = (double*)gray1d.data;
	byte* grayPu = (byte*)gray1u.data;
	for (int i = 0; i < N; i++){
		grayPu[i] = (img[i][0] + img[i][1] + img[i][2] + 1)/3;
		grayPd[i] = grayPu[i] * scale;
	}
}

void filter2D(CMat &src1d, Mat &dst1d, CMat &kernel1d)
{
	int W = src1d.cols, H = src1d.rows, sz = kernel1d.cols;
	CV_Assert(sz % 2 == 1 && sz == kernel1d.rows);
	int kNum = sz * sz, hSz = sz / 2;
	vector<Point> kPnts(kNum), dPnts(kNum);
	for (int r = -hSz, p = 0; r <= hSz; r++) for (int c = -hSz; c <= hSz; c++, p++)
		kPnts[p] = Point(c + hSz, r + hSz), dPnts[p] = Point(c, r);

	dst1d = Mat::zeros(H, W, CV_64F);
	int WM1 = W - 1, HM1 = H - 1;
	for (int r = 0; r < H; r++) {
		double* dst = dst1d.ptr<double>(r);
		for (int c = 0; c < W; c++){
			Point pnt(c, r); double s = 0;
			for (int k = 0; k < kNum; k++){
				Point p = pnt + dPnts[k];
				p.x = CLAMP(p.x, 0, WM1), p.y = CLAMP(p.y, 0, HM1);
				s += kernel1d.at<double>(kPnts[k]) * src1d.at<double>(p);
			}
			dst[c] = s;
		}
	}
}

// Prepare image statistics for feature extraction
void RegionFeature::setImg(CMat &img3u, CMat &img3f)
{
	_H = img3u.rows, _W = img3u.cols, _N = _H * _W;	
	Mat gray1u, gray1d;
	cvtColor(img3u, gray1u, CV_BGR2GRAY);	
	gray1u.convertTo(gray1d, CV_64F, 1/255.0);

	Mat l_labImg3f, l_hsvImg3f;
	cvtColor(img3f, l_labImg3f, CV_RGB2Lab); 
	cvtColor(img3f, l_hsvImg3f, CV_RGB2HSV); 

	Mat l_qRgbIm1i, l_qLabIm1i, l_qHsvIm1i;   // Index of each pixel for building histograms
	//CmColorQua::S_Quantize(img3f, l_qRgbIm1i, CmColorQua::S_BGR);
	//CmColorQua::S_Quantize(img3f, l_qLabIm1i, CmColorQua::S_LAB);
	//CmColorQua::S_Quantize(img3f, l_qHsvIm1i, CmColorQua::S_HSV);
	S_Quantize(img3f, l_qRgbIm1i, S_BGR);
	S_Quantize(l_labImg3f, l_qLabIm1i, S_LAB);
	S_Quantize(l_hsvImg3f, l_qHsvIm1i, S_HSV);

	CV_Assert(mlFilters1d.size() == DIM_TEX);
	vecM l_imText1d(DIM_TEX);
#pragma omp parallel for
	for (int i = 0; i < DIM_TEX; i++){
		filter2D(gray1d, l_imText1d[i], CV_64F, mlFilters1d[i], Point(0, 0), 0.0, BORDER_REPLICATE);
		normalize(l_imText1d[i], l_imText1d[i], 0, 10, NORM_MINMAX);
	}
	
	Mat l_texMax1d = l_imText1d[0].clone();
	Mat l_qTextIm1i = Mat::zeros(img3f.size(), CV_32S);	
	Mat l_qLbpIm1i = getLbp(gray1u);
	m_bgrLabHsv9d.create(_H, _W, CV_64FC(DIM_CLR)), m_qRgbLabHsvTextLbp5i.create(_H, _W, CV_32SC(DIM_HIST)), m_imTex15d.create(_H, _W, CV_64FC(DIM_TEX)); 
	_bgrLabHsv9d = (ColorFea*)(m_bgrLabHsv9d.data);
	_rgbLabHsvTextLbp5i = (HistFea*)(m_qRgbLabHsvTextLbp5i.data);
	_imTex15d = (TextureFea*)(m_imTex15d.data);
	const Vec3f *bgr = (Vec3f*)img3f.data, *lab = (Vec3f*)l_labImg3f.data, *hsv = (Vec3f*)l_hsvImg3f.data;
	const int *qRgb = (int*)l_qRgbIm1i.data, *qLab = (int*)l_qLabIm1i.data, *qHsv = (int*)l_qHsvIm1i.data;
	const int *qLbp = (int*)l_qLbpIm1i.data;
	int *qTex = (int*)l_qTextIm1i.data;	
	double *textMax = (double*)l_texMax1d.data;
#pragma omp parallel for
	for (int i = 0; i < _N; i++){
		_imTex15d[i][0] = textMax[i];
		for (int t = 1; t < DIM_TEX; t++){
			double tVal = ((double*)l_imText1d[t].data)[i];
			_imTex15d[i][t] = tVal;
			if (tVal > textMax[i])
				qTex[i] = t, textMax[i] = tVal;
		}
		_bgrLabHsv9d[i] = Vec<double, 9>(bgr[i][0], bgr[i][1], bgr[i][2], lab[i][0], lab[i][1], lab[i][2], hsv[i][0], hsv[i][1], hsv[i][2]);
		_rgbLabHsvTextLbp5i[i] = Vec<int, 5>(qRgb[i], qLab[i], qHsv[i], qTex[i], qLbp[i]);
	}
}

inline void normalizeVecL1(Mat &src1i, Mat &dst1f){src1i.convertTo(dst1f, CV_32F, 1.0/sum(src1i).val[0]);}
template <int D> void absDif(Vec<double, D> &v1, Vec<double, D> &v2, double *dst){ for (int i = 0; i < D; i++) dst[i] = abs(v1[i] - v2[i]); }
template <int D> void updateVariance(const Vec<double, D> &crntSample, const Vec<double, D> &avergeVal, Vec<double, D> &varData) {for (int i = 0; i < D; i++) varData[i] += sqr(crntSample[i] - avergeVal[i]); }
inline double compareHistogram(CMat &h1_1f, CMat &h2_1f) // To play as an alternative for the OpenCV version with bug
{
	CV_Assert(h1_1f.type() == CV_32F && h2_1f.type() == CV_32F && h1_1f.rows == 1 && h2_1f.rows == 1 && h1_1f.cols == h2_1f.cols); 
	double res = 0;
	const float *h1 = (float*)h1_1f.data, *h2 = (float*)h2_1f.data;
	for (int i = 0; i < h1_1f.cols; i++)
		res += sqr(h1[i] - h2[i])/(h1[i] + h2[i] + EPS);
	return res * 0.5; 
}

Mat RegionFeature::getFeature(CMat &regIdx1i, int numReg)
{
	int NumReg = numReg + 1; //Include the border region at the last one
	vector<ColorFea> regColor9d(NumReg), regColorVar9d(NumReg);
	vector<TextureFea> regText15d(NumReg), regTextVar15d(NumReg);
	vecD regAvgLbp(NumReg), regVarLbp(NumReg);

	Mat l_borderIdx1i_ = getBorderReg(Size(_W, _H), 0.0375); // None border is -1, border is 0
	const int *regIdx = (int*)regIdx1i.data, *bdIdx = (int*)l_borderIdx1i_.data;
	Mat regHists1i[DIM_HIST], regHists1f[DIM_HIST];
	int HistWidth[DIM_HIST] = {256, 256, 256, DIM_TEX, 256};// Rgb, Lab, Hsv, Text, Lbp
	// int HistWidth[DIM_HIST] = {512, 512, 512, DIM_TEX, 256};// Rgb, Lab, Hsv, Text, Lbp
	for (int i = 0; i < DIM_HIST; i++)
		regHists1i[i] = Mat::zeros(NumReg, HistWidth[i], CV_32S), regHists1f[i] = Mat(NumReg, HistWidth[i], CV_32F);

	// Collect color features, area feature; prepare for perimeter, histogram distance features and region weights
	vecD regArea(NumReg); // The last one if for counting border region
	vector<Point2d> meanPos(NumReg);
	Mat_<double> regNeighborWeight1d = Mat::zeros(numReg, numReg, CV_64F);
	Mat l_edgePoint1i(_H, _W, CV_32S); // -1 for edge point and 0 for none edge point
	int *edgePoint = (int*)l_edgePoint1i.data;
	memset(edgePoint, -1, sizeof(int)*_N);
	l_edgePoint1i(Rect(1, 1, _W - 2, _H - 2)) = cv::Scalar(0);
	for (int r = 0, p = 0; r < _H; r++){
		for (int c = 0; c < _W; c++, p++){
			int idx = regIdx[p];
			if (c > 0 && idx != regIdx[p-1]) {// Check left one for creating neighbor
				int neiIdx = regIdx[p-1];
				regNeighborWeight1d[idx][neiIdx] = regNeighborWeight1d[neiIdx][idx] = 1;
				edgePoint[p] = -1, edgePoint[p-1] = -1;
			}
			if (r > 0 && idx != regIdx[p - _W]) {// Check upper one for creating neighbor
				int neiIdx = regIdx[p - _W];
				regNeighborWeight1d[idx][neiIdx] = regNeighborWeight1d[neiIdx][idx] = 1;
				edgePoint[p] = -1, edgePoint[p - _W] = -1;
			}
			meanPos[idx] += Point2d(c, r);
			regAvgLbp[idx] += _rgbLabHsvTextLbp5i[p][4];

			for (int i = 0; i < 2; i++){ // Segmentation index for i == 0, border index for i == 1
				if (i == 1){
					if (bdIdx[p] == -1)
						continue;
					idx = numReg;
				}
				regArea[idx]++; // Add a new sample for this region
				regColor9d[idx] += _bgrLabHsv9d[p];
				regText15d[idx] += _imTex15d[p];
				const HistFea &histIdx = _rgbLabHsvTextLbp5i[p];
				for (int h = 0; h < DIM_HIST; h++)
					regHists1i[h].at<int>(idx, histIdx[h])++;
			}
		}
	}
	
	vecD regPerimeter(numReg); // Get region perimeter
	for (int p = 0; p < _N; p++)
		regPerimeter[regIdx[p]] += (edgePoint[p] != 0 ? 1 : 0);

	for (int i = 0; i < NumReg; i++){// Update average region colors, average texture values. normalize region histograms
		double crntRegArea = regArea[i];
		regColor9d[i] /= crntRegArea, regText15d[i] /= crntRegArea;
		meanPos[i].x /= _W * crntRegArea;
		meanPos[i].y /= _H * crntRegArea;
		regAvgLbp[i] /= crntRegArea;
		for (int h = 0; h < DIM_HIST; h++)
			regHists1i[h].row(i).convertTo(regHists1f[h].row(i), CV_32F, 1.0/sum(regHists1i[h].row(i)).val[0]);
	} 

	// ================================= global contrast, added by Huaizu ==============================================
	vecD regNeighborArea(numReg);
	for (int i = 0; i < numReg; i++){		
		double* regNeighborWeight = regNeighborWeight1d.ptr<double>(i);
		double sumW = 0;
		for (int j = 0; j < numReg; j++){
			sumW += regNeighborWeight[j];
		}
		regNeighborArea[i] = sumW;
	}

	static const double sigmaDist = 0.4;
	for( int i = 0; i < numReg; ++i )
	{
		const Point2d &centroid = meanPos[i];
		for( int j = 0; j < numReg; ++j )
		{
			if( i <= j )
				// regNeighborWeight1d[i][j] = exp( -pntSqrDist(centroid, meanPos[j])/ (2.0 * sigmaDist * sigmaDist) );
				regNeighborWeight1d[i][j] = exp( -pntSqrDist(centroid, meanPos[j]) / sigmaDist );
			else
				regNeighborWeight1d[i][j] = regNeighborWeight1d[j][i];
		}
	}
	// =================================================================================================================

	
	// Update regNeighborWeight1d & get feature contrast
	Mat feaCtrstNd = Mat::zeros(numReg, NumReg, CV_64FC(DIM_CTRST));
	for (int i = 0; i < numReg; i++){		
		double* regNeighborWeight = regNeighborWeight1d.ptr<double>(i);
		double sumW = 0;
		for (int j = 0; j < numReg; j++){
			regNeighborWeight[j] *= regArea[j];
			sumW += regNeighborWeight[j];
		}
		for (int j = 0; j < numReg; j++)
			regNeighborWeight[j] /= sumW;

		// Get feature contrast
		for (int j = i+1; j < NumReg; j++){ // Contrast with all other and background one (last column)
			CtrstVal &cVal29d = feaCtrstNd.at<CtrstVal>(i, j);
			double *cVal = (double*)(&cVal29d);
			absDif(regColor9d[i], regColor9d[j], cVal);
			absDif(regText15d[i], regText15d[j], cVal + DIM_CLR);
			cVal += DIM_CLR + DIM_TEX;
			for (int h = 0; h < DIM_HIST; h++)// compareHist(regHists1f[h].row(i), regHists1f[h].row(j), CV_COMP_CHISQR);
				cVal[h] = compareHistogram(regHists1f[h].row(i), regHists1f[h].row(j)); 
			if (j < numReg)
				feaCtrstNd.at<CtrstVal>(j, i) = cVal29d;
		}
	}	

	//std::ofstream out( "w.bin", std::ios::binary );
	//out.write( (char*)(&numReg), sizeof(int) );
	//out.write( (char*)(&numReg), sizeof(int) );
	//for( int i = 0; i < numReg; ++i )
	//{
	//	for( int j = 0; j < numReg; ++j )
	//	{
	//		out.write( (char*)(&regNeighborWeight1d[i][j]), sizeof(double) );
	//	}
	//}
	//out.close();

	// Update variance features
	for (int p = 0; p < _N; p++){
		int idx = regIdx[p];
		updateVariance(_bgrLabHsv9d[p], regColor9d[idx], regColorVar9d[idx]);
		updateVariance(_imTex15d[p], regText15d[idx], regTextVar15d[idx]);
		double v1 = _rgbLabHsvTextLbp5i[p][4], v2 = regAvgLbp[idx];
		regVarLbp[idx] += sqr(_rgbLabHsvTextLbp5i[p][4] - regAvgLbp[idx]);
	}

	Mat fea1Vald = Mat::zeros(numReg, DIM_F, CV_64F);	
	vector<Vec4d> regBound = getRegBound(regIdx1i, regArea); // [left, top, right, bottom]

	for (int i = 0; i < numReg; i++){
		// Contrast of each region to its neighbor regions and background region
		double* w = regNeighborWeight1d.ptr<double>(i);  
		CtrstVal regCtrst, &backCtrst = feaCtrstNd.at<CtrstVal>(i, numReg);
		for (int j = 0; j < numReg; j++)
			if (w[j] > EPS)
				regCtrst += feaCtrstNd.at<CtrstVal>(i, j) * w[j];

		// Variance based features
		regColorVar9d[i] /= max(regArea[i] - 1, 1.0);
		regTextVar15d[i] /= max(regArea[i] - 1, 1.0);
		regVarLbp[i] /= max(regArea[i] - 1, 1.0);
		double *feaV = fea1Vald.ptr<double>(i);

#ifdef USE_HUAIZU_ORDER
		const int idxMaping[DIM_CTRST] = {2, 1, 0, 4, 5, 6, 8, 9, 10, // 9d Colors: B, G, R, L, a, b, H, S, V,
			12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,   // 15d texture
			3, 7, 11, 27, 28  // 5D hist: rgb, lab, hsv, text, lbp
		};// Index mapping for color and histogram contrast
		for (int j = 0; j < DIM_CTRST; j++)
			feaV[idxMaping[j]] = regCtrst[j], feaV[idxMaping[j]+DIM_CTRST] = backCtrst[j];
#else
		memcpy(feaV, &regCtrst, sizeof(CtrstVal));
		memcpy(feaV + DIM_CTRST, &backCtrst, sizeof(CtrstVal));
#endif // USE_HUAIZU_ORDER

		// 35d region property features. These index values in the following are w.r.t. I_REG_PROP, and represented by [*]
		feaV += DIM_CTRST*2;
		memcpy(feaV, &meanPos[i], sizeof(Point2d)); // [0, 1]: 2d region centroid (x, y)
		Vec4d &regB = regBound[i];
		memcpy(feaV + 2, &regB, sizeof(Vec4d)); // [2, 3, 4, 5]: 4d border feature (left, top, right, bottom)
		feaV[6] = regPerimeter[i]/(_W + _H); // [6]: 1d perimeter feature
		feaV[7] = (regB[2] - regB[0]) / (regB[3] - regB[1] + EPS);// [7]: 1d aspect ratio feature: (RIGHT - LEFT) / (BOTTOM - TOP + eps)
		memcpy(feaV + 8, &regColorVar9d[i][0], sizeof(ColorFea));
		memcpy(feaV + 17, &regTextVar15d[i][0], sizeof(TextureFea));

		// Follow features are multiplied with certain values just for having them have ralply the same data range
		feaV[32] = 0.0001 * regVarLbp[i]; // [32]: 1d LBP variance feature
		feaV[33] = 100 * regArea[i]/_N; // [33]: 1d area feature
		feaV[34] = 10 * regNeighborArea[i] / _N; // [34]: 1d area of neighbor feature 
	}
	return fea1Vald;
}

// Return a vector with Vec4d elements [left, top, right, bottom]
vector<Vec4d> RegionFeature::getRegBound(CMat &regIdx1i, const vecD &regArea)
{
	CV_Assert(regIdx1i.isContinuous());
	const int W = regIdx1i.cols, H = regIdx1i.rows, regNum = regArea.size() - 1;
	vector<Vec4d> regBound(regNum);
	vecI idxTop10(regNum), idxTop90(regNum), crntIdx(regNum);
	for (int i = 0; i < regNum; i++)
		idxTop10[i] = cvRound(regArea[i] * 0.1), idxTop90[i] = cvRound(regArea[i] * 0.9);
	int* regIdx = (int*)regIdx1i.data;
	for (int r = 0, p = 0; r < H; r++) for (int c = 0; c < W; c++, p++){ // Get top and bottom index
		int idx = regIdx[p];
		if (crntIdx[idx] == idxTop10[idx]) // have the top value
			regBound[idx][1] = r;
		if (crntIdx[idx] == idxTop90[idx]) // have the top value
			regBound[idx][3] = r;
		crntIdx[idx]++;
	}
	memset(&crntIdx[0], 0, regNum * sizeof(int));
	for (int c = 0; c < W; c++) for (int r = 0, p = c; r < H; r++, p+= W){
		int idx = regIdx[p];
		if (crntIdx[idx] == idxTop10[idx]) // have the left value
			regBound[idx][0] = c;
		if (crntIdx[idx] == idxTop90[idx]) // have the right value
			regBound[idx][2] = c;
		crntIdx[idx]++;		
	}
	for (int i = 0; i < regNum; i++)
		regBound[i] = Vec4d(regBound[i][0]/W, regBound[i][1]/H, regBound[i][2]/W, regBound[i][3]/H);
	return regBound;
}

Mat RegionFeature::getLbp(CMat &grayImg1u)
{
	Mat lbp1i(grayImg1u.size(), CV_32S);
	const int _w = grayImg1u.cols, _h = grayImg1u.rows;
//#pragma omp parallel for
	for (int r = 0; r < _h; r++){
		const byte* imVal = grayImg1u.ptr<byte>(r);
		int* lbpVal = lbp1i.ptr<int>(r);
		for (int c = 0; c < _w; c++){
			byte crnt = imVal[c];
			int lbpCrnt = 0;
			Point pnt(c, r);
			for (int i = 0; i < 8; i++)	{
				Point pntN = pnt + DIRECTION8[i];
				if (CHK_IND(pntN) && grayImg1u.at<byte>(pntN) > crnt)
					lbpCrnt |= (1 << i); 
			}
			lbpVal[c] = lbpCrnt;
			CV_Assert(lbpCrnt >= 0 && lbpCrnt < 256);
		}
	}
	return lbp1i;
}


Mat RegionFeature::getBorderReg(Size imgSz, double borderRatio)
{
	int borderW = cvRound(max(imgSz.width, imgSz.height) * borderRatio);
	int borderH = borderW;
	Mat regidx(imgSz, CV_32S);
	memset(regidx.data, -1, imgSz.width*imgSz.height*sizeof(int));
	regidx.rowRange(0, borderH).setTo(0);
	regidx.rowRange(imgSz.height - borderH, imgSz.height).setTo(0);
	regidx.colRange(0, borderW).setTo(0);
	regidx.colRange(imgSz.width - borderW, imgSz.width).setTo(0);
	return regidx;
}

void RegionFeature::S_Quantize(CMat& img, Mat &idx1i, int method)
{
	CV_Assert(method >= 0 && method < S_Q_NUM && img.data != NULL && img.type() == CV_32FC3);
	S_QUANTIZE_FUNC SQ_Function = sqFuns[method];

	idx1i.create(img.size(), CV_32S);
	for (int r = 0; r < img.rows; r++)	{
		const Vec3f * imgD = img.ptr<Vec3f>(r);
		int *idx = idx1i.ptr<int>(r);
		for (int c = 0; c < img.cols; c++)
			idx[c] = (*SQ_Function)(imgD[c]);
	}
}

int RegionFeature::SQ_BGR(const Vec3f &c)
{
	return (int)(c[0]*5.9999f) * 36 + (int)(c[1]*5.9999f) * 6 + (int)(c[2]*5.9999f);
}

int RegionFeature::SQ_HSV(const Vec3f &c)
{
	const float S_MIN_HSV = 0.1f;
	float h(c[0]/360.001f), s(c[1]/1.001f), v(c[2]/1.001f);

	int result;
	if (s < S_MIN_HSV) // 240 ... 255
		result = 240 + (int)(v * 16.f); 
	else{
		int ih, is, iv;
		ih = (int)(h * 15.f); //0..14
		is = (int)((s - S_MIN_HSV)/(1 - S_MIN_HSV) * 4.f); //0..3
		iv = (int)(v * 4.f); //0..3
		result = ih * 16 + is * 4 + iv; // 0..239

		CV_Assert(ih < 15 && is < 4 && iv < 4);
	}

	return result;
}

int RegionFeature::SQ_Lab(const Vec3f &c)
{
	float L(c[0] / 100.0001f), a((c[1] + 127) / 254.0001f), b((c[2] + 127) / 254.0001f);
	int iL = (int)(L * 5), ia = (int)(a * 7), ib = (int)(b * 7);
	CV_Assert(iL >= 0 && ia >= 0 && ib >= 0 && iL < 5 && ia < 7 && ib < 7);
	return iL + ia * 5 + ib * 35;
}
