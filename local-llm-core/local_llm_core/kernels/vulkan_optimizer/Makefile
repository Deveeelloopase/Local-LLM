all: matmul_1bit.spv

matmul_1bit.spv: matmul_1bit.comp
	glslc -o $@ $<

clean:
	rm -f matmul_1bit.spv
