//#include "stdafx.h"
#include "segment-graph.h"
#include "segment-image.h"

// dissimilarity measure between pixels
static inline float diff(CMat &img3f, int x1, int y1, int x2, int y2)
{
	const Vec3f &p1 = img3f.at<Vec3f>(y1, x1);
	const Vec3f &p2 = img3f.at<Vec3f>(y2, x2);
	return sqrt(sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1]) + sqr(p1[2] - p2[2]));
}


universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c) {
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	universe *u = new universe(num_vertices);

	// init thresholds
	float *threshold = new float[num_vertices];
	for (int i = 0; i < num_vertices; i++)
		threshold[i] = THRESHOLD(1, c);

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < num_edges; i++) {
		edge *pedge = &edges[i];

		// components connected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b) {
			if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) {
				u->join(a, b);
				a = u->find(a);
				threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
			}
		}
	}

	// free up
	delete threshold;
	return u;
}

int SegmentImage(const image<RGB_f> *im, image<int> *segIdx, float sigma, float c, int min_size)
{
	int width = im->width();
	int height = im->height();
	image<float> *r = new image<float>(width, height);
	image<float> *g = new image<float>(width, height);
	image<float> *b = new image<float>(width, height);

	// smooth each color channel  
	for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) {
		imRef(r, x, y) = imRef(im, x, y).r;
		imRef(g, x, y) = imRef(im, x, y).g;
		imRef(b, x, y) = imRef(im, x, y).b;
	}
	image<float> *smooth_r = smooth(r, sigma);
	image<float> *smooth_g = smooth(g, sigma);
	image<float> *smooth_b = smooth(b, sigma);
	delete r;
	delete g;
	delete b;

	// build graph
	edge *edges = new edge[width*height * 4];
	int num = 0;
	for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) {
		if (x < width - 1) {
			edges[num].a = y * width + x;
			edges[num].b = y * width + (x + 1);
			edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y);
			num++;
		}

		if (y < height - 1) {
			edges[num].a = y * width + x;
			edges[num].b = (y + 1) * width + x;
			edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y + 1);
			num++;
		}

		if ((x < width - 1) && (y < height - 1)) {
			edges[num].a = y * width + x;
			edges[num].b = (y + 1) * width + (x + 1);
			edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y + 1);
			num++;
		}

		if ((x < width - 1) && (y > 0)) {
			edges[num].a = y * width + x;
			edges[num].b = (y - 1) * width + (x + 1);
			edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x + 1, y - 1);
			num++;
		}
		
	}
	delete smooth_r;
	delete smooth_g;
	delete smooth_b;
		
	universe *u = segment_graph(width*height, num, edges, c); // segment

	// post process small components
	for (int i = 0; i < num; i++) {
		int a = u->find(edges[i].a), b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete[] edges;
	map<int, int> marker;
	// = new image<int>(width, height);
	int idxNum = 0;
	for (int y = 0; y < height; y++) {
		int *imgIdx = segIdx->access[y];
		for (int x = 0; x < width; x++) {
			int comp = u->find(y * width + x);
			if (marker.find(comp) == marker.end())
				marker[comp] = idxNum++;
			imgIdx[x] = marker[comp];
		}
	}
	assert(idxNum == u->num_sets());
	delete u;
	return idxNum;
}

int SegmentImage(CMat &_src3f, Mat &pImgInd, float sigma, float c, int min_size)
{
	CV_Assert(_src3f.type() == CV_32FC3);
	int width(_src3f.cols), height(_src3f.rows);
	pImgInd.create(height, width, CV_32S);
	image<RGB_f> *im = new image<RGB_f>(width, height, _src3f.data);
	image<int> *regIdx = new image<int>(width, height, pImgInd.data);
	int regNum = SegmentImage(im, regIdx, sigma, c, min_size);
	im->data = NULL; regIdx->data = NULL;
	delete im; delete regIdx;
	return regNum;
}