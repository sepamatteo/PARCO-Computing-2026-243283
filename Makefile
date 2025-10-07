# Compiler settings
CXX = g++
CC = gcc

CXXFLAGS = -O2 -std=c++17 -Wall
CFLAGS = -O2 -Wall

# Targets
TARGET_COO = spmv_coo
TARGET_CSR = spmv_csr

# Source files
SRCS_COO = src/spmv_coo.cpp src/mmio.c
SRCS_CSR = src/spmv_csr.cpp src/mmio.c

# Default target builds both
all: $(TARGET_COO) $(TARGET_CSR)

# Build COO executable
$(TARGET_COO): $(SRCS_COO)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build CSR executable
$(TARGET_CSR): $(SRCS_CSR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean build artifacts
clean:
	rm -f $(TARGET_COO) $(TARGET_CSR)
	rm -f src/*.o

.PHONY: all clean
