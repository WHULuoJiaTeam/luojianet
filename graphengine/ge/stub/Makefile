inc_path := $(shell pwd)/graphengine/inc/external/
out_path := $(shell pwd)/out/ge/lib64/stub/
stub_path := $(shell pwd)/graphengine/ge/stub/

mkdir_stub := $(shell mkdir -p $(out_path))
local_stub := $(shell $(HI_PYTHON) $(stub_path)/gen_stubapi.py $(inc_path) $(out_path))
