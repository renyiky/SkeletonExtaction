VPATH := include:src
ROOT := $(shell pwd)
INC_DIR := $(ROOT)/include
SRC_DIR := $(ROOT)/src
OBJ_DIR := $(SRC_DIR)/obj

SRC_CODE := $(notdir $(wildcard $(SRC_DIR)/*.cpp))
OBJ := $(SRC_CODE:%.cpp=%.o)

CXX := clang++
CXXFLAGS := -std=c++11 -I $(INC_DIR) `pkg-config --cflags --libs opencv4` -w


app:makeOBJDIR $(OBJ)
	@echo Linking...
	@$(CXX) -o app $(wildcard $(OBJ_DIR)/*.o) $(CXXFLAGS)

makeOBJDIR:
	@mkdir -p $(OBJ_DIR)

%.o:%.cpp
	@echo Compiling $(notdir $<)...
	@$(CXX) -c $< -o $(OBJ_DIR)/$@ $(CXXFLAGS) 

.PHONY: clean cleanall cleanres

cleanall: clean cleanres

clean:
	@rm -rf app $(OBJ_DIR)
	@echo App and objs cleaned.

cleanres:
	@rm -rf results/*.png
	@echo Results cleaned.