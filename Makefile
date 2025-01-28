STATS = sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.max 
TARGET = swizzle
all:
	@cd build && make
run: 
	@./build/$(TARGET)
ncu:
	ncu --metrics $(STATS) build/$(TARGET)