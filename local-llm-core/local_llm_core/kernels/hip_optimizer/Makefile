all: ternary_matmul.so

ternary_matmul.so: hip_matmul.cpp
	hipcc -O3 -shared -fPIC $< -o $@

clean:
	rm -f ternary_matmul.so
	