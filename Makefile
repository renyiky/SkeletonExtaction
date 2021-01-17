VPATH := include:src:src/comparisonAlgs:src/synthesizer:src/evaluation

SRC_CPP := $(notdir $(wildcard src/*.cpp))
COM_CPP	:= $(notdir $(wildcard src/comparisonAlgs/*.cpp))

OBJ := $(SRC_CPP:%.cpp=%.o) $(COM_CPP:%.cpp=%.o)

CXX_STD := -std=c++17
CXX_INC_DIR := -Iinclude
CXX := clang++
CXXFLAGS := -MMD `pkg-config --cflags --libs opencv4` -w $(CXX_STD) $(CXX_INC_DIR)

app:$(OBJ)
	@echo Linking...
	@$(CXX) -o $@ $^ $(CXXFLAGS)
	@echo Done.

-include $(OBJ:.o=.d)

# compile main app
%.o:src/%.cpp
	@echo Compiling $<...
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

# compile comparison algs
%.o:src/comparisonAlgs/%.cpp
	@echo Compiling $<...
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

# for evaluator
ev:
	@echo Generating Evaluator...
	@$(CXX) -o $@ src/evaluation/evaluation.cpp $(CXXFLAGS) 
	@echo Done.

# for synthesizer
syn:
	@echo Generating Synthesizer...
	@$(CXX) -o $@ src/synthesizer/synthesizer.cpp $(CXXFLAGS) 
	@echo Done.

.PHONY: clean cleanres do ev syn

clean:
	@rm -rf app *.o *.d
	@echo App and objs cleaned.

cleanres:
	@rm -rf results/*.png
	@rm -rf results/visualization/*.png
	@rm -rf experimentsMaterial/results/*.png
	@echo Results cleaned.

do: cleanres
	@make