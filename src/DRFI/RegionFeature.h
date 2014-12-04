#pragma once

class RegionFeature
{
private: 
	static const int DIM_F = 93; // Total dimension of region features: 29 * 2 + 35
	static const int DIM_CLR = 9, DIM_TEX = 15, DIM_HIST = 5; // dimension of the texture, color and histogram features
	static const int DIM_CTRST = DIM_TEX + DIM_HIST + DIM_CLR; // 29 dimension of contrast features
	typedef Vec<double, DIM_CTRST> CtrstVal;
	typedef Vec<double, DIM_CLR> ColorFea; // 9D color feature
	typedef Vec<double, DIM_TEX> TextureFea; // 15D texture feature
	typedef Vec<int, DIM_HIST> HistFea;// 15d histogram feature

public: // Interface functions	
	static void setFilter(CMat &mlFilters15d); // Needs to be called before other functions

	void setImg(CMat &img3u, CMat &img3f); // Prepare image statistics for feature extraction

	Mat getFeature(CMat &regIdx1i, int numReg);  // Each row is a 93d feature for an region.

public: // Helper functions

	// Return a vector with Vec4d elements [left, top, right, bottom]
	static vector<Vec4d> getRegBound(CMat &regIdx1i, const vecD &regArea);

	static double HistDistance( const CMat &h1, const CMat &h2, int type = 0 );

	static Mat getLbp(CMat &grayImg1f);

	// Return a int matrix with 0 for border pixel and -1 for others
	static Mat getBorderReg(Size imgSz, double borderRatio = 0.0375);

protected:
	enum {S_BGR, S_HSV, S_LAB, D_BGR};
	typedef int (*S_QUANTIZE_FUNC)(const Vec3f &c);  
	static void S_Quantize(CMat& img3f, Mat &idx1i, int method = S_BGR);
	static const int S_Q_NUM = 3;
	static const S_QUANTIZE_FUNC sqFuns[S_Q_NUM];

	// Static quantization and recover without prior color statistics
	static int SQ_BGR(const Vec3f &c);		// B,G,R[0,1] --> [0, 215]
	static int SQ_HSV(const Vec3f &c);		// H[0,360], S,V[0,1] ---> [0, 255]
	static int SQ_Lab(const Vec3f &c);		// L[0, 100], a, b [-127, 127] ---> [0, 244]

private: // for image information
	int _H, _W, _N; // Height, Width, Number of pixels for the input image
	
	// Data terms and corresponding pointers
	Mat m_bgrLabHsv9d, m_qRgbLabHsvTextLbp5i, m_imTex15d; // Data terms
	ColorFea *_bgrLabHsv9d;
	HistFea *_rgbLabHsvTextLbp5i;
	TextureFea *_imTex15d;

private:
	static vecM mlFilters1d;
};


