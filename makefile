# Makefile 

CC = gcc
CFLAGS = -Wall -Werror
LDFLAGS = -lm

# Directories
SRC_DIR = src
LIB_DIR = lib
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Libraries
LIB_FILES = $(LIB_DIR)/matrix_lib.h $(LIB_DIR)/neuronal_lib.h

# Executable
EXECUTABLE = $(BIN_DIR)/simple_neuronal_implementation

# Targets
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(LIB_FILES)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f $(EXECUTABLE)

.PHONY: all clean
