"""Offline routing benchmark: python -m benchmarks.bench_routing.

Reports median times after JIT warmup and peak allocations visible to
tracemalloc. The latter excludes Numba's native allocator and is not process
peak RSS. There are no pass/fail timing thresholds.
"""

import gc
import json
import statistics
import time
import tracemalloc

import numpy as np
import pandas as pd

from iduedu import UrbanGraph


def main():
    """Compare separate calculations, joint eager results and joint lazy results."""
    side = 40
    edges = []
    for y in range(side):
        for x in range(side):
            node = y * side + x
            for target in ([node + 1] if x + 1 < side else []) + ([node + side] if y + 1 < side else []):
                edges.append((node, target, 1.0, 10.0, None))
    graph = UrbanGraph(
        pd.DataFrame(index=range(side * side)),
        pd.DataFrame(edges, columns=["u", "v", "time_min", "length_meter", "geometry"]),
        is_multigraph=False,
        is_directed=False,
    )
    rng = np.random.default_rng(28)
    params = dict(
        origins_nodes=rng.choice(side * side, 60, replace=False),
        destination_nodes=rng.choice(side * side, 100, replace=False),
        cutoff=25,
        max_workers=1,
    )

    def separate():
        return graph.od_matrix(**params), graph.od_matrix(**params, output="paths")

    operations = {
        "distances": lambda: graph.od_matrix(**params),
        "paths": lambda: graph.od_matrix(**params, output="paths"),
        "separate": separate,
        "both_eager": lambda: graph.od_matrix(**params, output="both"),
        "both_lazy": lambda: graph.od_matrix(**params, output="both", path_storage="lazy"),
    }
    timings = {}
    for name, operation in operations.items():
        operation()  # JIT and adjacency warmup, outside measurements.
        elapsed = []
        for _ in range(5):
            gc.collect()
            started = time.perf_counter()
            result = operation()
            elapsed.append(time.perf_counter() - started)
            del result
        tracemalloc.start()
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del result
        timings[name] = {
            "median_ms": round(statistics.median(elapsed) * 1000, 2),
            "python_peak_mib": round(peak / 1024**2, 3),
        }
    print(
        json.dumps(
            {
                "nodes": len(graph.nodes_gdf),
                "origins": 60,
                "destinations": 100,
                "cutoff": 25,
                "workers": 1,
                "results": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
