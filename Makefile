# Compiler
CC = gcc

# Compiler flags (mavx2 is for simd, fma is for fused mulacc)
# CFLAGS = -O2 -Wall -Wextra -std=c11 -mavx2 -mfma
CFLAGS = -O3 -Wall -Wextra -std=c11 -mavx2 -mfma -march=native -pthread

# Source and target files
SRC = src/main.c
TARGET = build/main

# Rule to build the target
$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< -o $@

# Clean rule to remove the build output
clean:
	rm -f $(TARGET)
