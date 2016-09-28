set(SOURCE_FILES
    "common.h"
    "common.cu"
    "cpu.h"
    "cpu.cu"
    "naive.h"
    "naive.cu"
    "efficient.h"
    "efficient.cu"
    "real_efficient.h"
    "real_efficient.cu"
    "thrust.h"
    "thrust.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_50
    )
