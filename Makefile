CXX ?= g++
CC ?= gcc
CFLAGS = -O3 -fopenmp -std=c++0x
INCS = -Isrc/CmLib -Isrc/CmLib/EfficientGraphBased -Isrc/DRFI -Isrc/RF_Reg_C
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui

all: drfi

drfi: cmlib rforest saliency segmentation src/MainDRFI.cpp
	$(CXX) $(CFLAGS) $(INCS) -o DRFI src/MainDRFI.cpp CmDefinition.o cokus.o reg_RF.o RegionFeature.o SalDRFI.o segment-image.o $(LIBS)
	rm -f *~ *.o

cmlib: src/CmLib/CmDefinition.cpp
	$(CXX) $(CFLAGS) -c src/CmLib/CmDefinition.cpp

segmentation: src/CmLib/EfficientGraphBased/segment-image.cpp cmlib
	$(CXX) $(CFLAGS) $(INCS) -c src/CmLib/EfficientGraphBased/segment-image.cpp CmDefinition.o

rforest: src/RF_Reg_C/cokus.cpp src/RF_Reg_C/reg_RF.cpp	
	$(CXX) $(CFLAGS) -c src/RF_Reg_C/cokus.cpp
	$(CXX) $(CFLAGS) -c src/RF_Reg_C/reg_RF.cpp

saliency: src/DRFI/RegionFeature.cpp src/DRFI/SalDRFI.cpp
	$(CXX) $(CFLAGS) $(INCS) -c src/DRFI/RegionFeature.cpp CmDefinition.o
	$(CXX) $(CFLAGS) $(INCS) -c src/DRFI/SalDRFI.cpp CmDefinition.o

clean:
	rm -f *~ CmDefinition.o segment-image.o cokus.o reg_RF.o RegionFeature.o SalDRFI.o DRFI
