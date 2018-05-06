# stop echo of command on termianl
.SILENT: all run debug clean
.PHONY: clean

CXX = g++ -Wextra -std=c++11
#Compilation flag
CXXFLAGS = `pkg-config --cflags opencv`
#Linking flag
LDFLAGS = -Iusr/local/include/opencv2 -lopencv_core -lopencv_imgcodecs
# Executable file
TARGET = main
#source files
SOURCE = $(TARGET).cpp io.cpp calculations.cpp

#
all : $(SOURCE)
	echo "Building code ... "
	$(CXX) -pg -o $(TARGET)".out" $(SOURCE) $(CXXFLAGS) $(LDFLAGS)
	echo "Build complete!! Run ./"$(TARGET) "to execute the code."

# to print debug messages
debug : $(SOURCE)
	echo "Building code in debug mode... "
	$(CXX) -DDEBUG -o $(TARGET)".out" $(SOURCE) $(CXXFLAGS) $(LDFLAGS)
	echo "Build complete!! Run ./"$(TARGET) "to execute the code."

clean:
	echo "Cleaning Up ... "
	rm -f *.o *.out *~ $(TARGET)
	echo "Done"

run:
	./$(TARGET)".out" ./input/training.txt ./input/test_full.txt 12