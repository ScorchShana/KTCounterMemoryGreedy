#pragma once

#include "HistogramCommon.h"

// Loads metadata, parses the final high/low group files and writes the TSV
// histogram. Failures throw after the AIO reader threads are released.
int run_histogram_tool(Options options);
