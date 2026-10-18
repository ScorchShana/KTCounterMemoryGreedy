#pragma once

#include "HistogramCommon.h"

// Loads metadata, selects the execution path and writes the TSV histogram.
// Failures throw after all running workers and input readers are released.
int run_histogram_tool(Options options);
