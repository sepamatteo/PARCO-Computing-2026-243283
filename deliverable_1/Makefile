# Compiler settings
CXX = g++
CC = gcc

CXXFLAGS = -O3 -std=c++11 -Wall -fopenmp -ffast-math -march=native
CFLAGS = -O3 -Wall

OUTPUT_DIR = outputs

# Full paths to executables
TARGET_COO = $(OUTPUT_DIR)/spmv_coo
TARGET_CSR = $(OUTPUT_DIR)/spmv_csr
TARGET_PARALLEL_CSR = $(OUTPUT_DIR)/parallel_spmv_csr

# Source files
SRCS_CPP_COO = src/spmv_coo.cpp
SRCS_CPP_CSR = src/spmv_csr.cpp
SRCS_CPP_PAR_CSR = src/parallel_spmv_csr.cpp
SRCS_C = src/mmio.c

# Object files
OBJS_CPP_COO = $(SRCS_CPP_COO:.cpp=.o)
OBJS_CPP_CSR = $(SRCS_CPP_CSR:.cpp=.o)
OBJS_CPP_PAR_CSR = $(SRCS_CPP_PAR_CSR:.cpp=.o)
OBJS_C = $(SRCS_C:.c=.o)

# Default target builds all
all: $(OUTPUT_DIR) $(TARGET_COO) $(TARGET_CSR) $(TARGET_PARALLEL_CSR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(TARGET_COO): $(OBJS_CPP_COO) $(OBJS_C) | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_CSR): $(OBJS_CPP_CSR) $(OBJS_C) | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^
	
$(TARGET_PARALLEL_CSR): $(OBJS_CPP_PAR_CSR) $(OBJS_C) | $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Shortcut targets for easy make calls
spmv_coo: $(TARGET_COO)
spmv_csr: $(TARGET_CSR)
spmv_par_csr: $(TARGET_PARALLEL_CSR)

clean:
	rm -f $(OBJS_CPP_COO) $(OBJS_CPP_CSR) $(OBJS_C) $(TARGET_COO) $(TARGET_CSR)

.PHONY: all clean spmv_coo spmv_csr
