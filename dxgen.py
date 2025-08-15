#!/usr/bin/env python3
"""
DX-Onto SPARQL Benchmark Suite (rdflib)

What it does
------------
- Builds synthetic RDF graphs of varying sizes: 10, 50, 100, 200, 500, 1000 projects
- For each size, runs 5 trials (fresh graph each time, new seed per trial)
- Executes 3 SPARQL queries representative of your manuscript:
    Q1: simple project-scoped multimedia retrieval
    Q2: cross-project retrieval with multiple constraints
    Q3: aggregation by technology
- Times each query (median & p95 over N runs with warmups)
- Writes:
    1) per-trial results  -> CSV (--out)
    2) aggregated summary -> CSV (--summary)

Usage
-----
  pip install rdflib
  python dxonto_sparql_benchsuite.py \
      --sizes 10 50 100 200 500 1000 \
      --trials 5 \
      --runs 20 \
      --warmups 3 \
      --docs-min 2 \
      --docs-max 5 \
      --seed 42 \
      --out dxonto_suite_results.csv \
      --summary dxonto_suite_summary.csv
"""

import argparse
import random
import time
import statistics
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

from rdflib import Graph, Namespace, Literal, RDF, URIRef


# --------------------
# Namespaces & constants
# --------------------
DX = Namespace("http://example.com/dx#")
SOUTH = {"Yala", "Songkhla", "Pattani", "Satun"}


# --------------------
# Data generation
# --------------------
def build_graph(num_projects: int = 1000,
                docs_per_project: Tuple[int, int] = (2, 5),
                seed: int = 42) -> Graph:
    """Create synthetic DX data in an rdflib Graph."""
    random.seed(seed)
    g = Graph()
    g.bind("dx", DX)

    # Classes
    DXProject      = DX.DXProject
    DxResourceType = DX.DxResourceType
    TechProduct    = DX.TechProduct
    ProjectDetail  = DX.ProjectDetail
    ProjectResult  = DX.ProjectResult
    DXDimension    = DX.DXDimension
    KPI            = DX.KPI

    # Properties
    hasResource      = DX.hasResource
    hasFileType      = DX.hasFileType
    hasFileName      = DX.hasFileName
    projectName      = DX.projectName
    hasProjectDetail = DX.hasProjectDetail
    location         = DX.location
    hasTechProduct   = DX.hasTechProduct
    productName      = DX.productName
    hasProjectResult = DX.hasProjectResult
    isProjectResult  = DX.isProjectResult
    hasKPI           = DX.hasKPI
    kpiName          = DX.kpiName
    kpiValue         = DX.kpiValue
    hasDimension     = DX.hasDimension
    dimensionDesc    = DX.dimensionDescription
    hasSector        = DX.hasSector
    isInPhase        = DX.isCurrentlyInPhase

    provinces = ["Yala", "Songkhla", "Satun", "Phuket", "Chiang Mai", "Bangkok", "Pattani"]
    tech_products = ["Drone", "Cloud CRM", "IoT Sensor", "ERP", "AI Vision", "RPA Bot"]
    sectors = ["tourism", "education", "agriculture", "public safety", "health", "transport"]
    dimensions = ["innovation", "informational", "cultural", "financial", "security", "quality"]
    phases = ["Digitization", "Digitalization", "DigitalTransformation"]
    filetypes = ["image", "word", "pdf", "pptx"]

    def uri(kind: str, i: int, j: Optional[int] = None) -> URIRef:
        return URIRef(f"{DX}{kind}_{i}" + (f"_{j}" if j is not None else ""))

    for i in range(num_projects):
        p = uri("Project", i)
        g.add((p, RDF.type, DXProject))
        pname = f"Project_{i}"
        g.add((p, projectName, Literal(pname)))

        # Detail
        d = uri("Detail", i)
        g.add((d, RDF.type, ProjectDetail))
        prov = random.choice(provinces)
        g.add((d, location, Literal(prov)))
        g.add((p, hasProjectDetail, d))

        # Sector
        sec = random.choice(sectors)
        g.add((p, hasSector, Literal(sec)))

        # Tech
        t = uri("Tech", i)
        g.add((t, RDF.type, TechProduct))
        tname = random.choice(tech_products)
        g.add((t, productName, Literal(tname)))
        g.add((p, hasTechProduct, t))

        # Result
        r = uri("Result", i)
        g.add((r, RDF.type, ProjectResult))
        success = "successful" if random.random() < 0.65 else "unsuccessful"
        g.add((r, isProjectResult, Literal(success)))
        g.add((p, hasProjectResult, r))

        # KPI
        k = uri("KPI", i)
        g.add((k, RDF.type, KPI))
        if random.random() < 0.5:
            g.add((k, kpiName, Literal("profit_increase")))
            g.add((k, kpiValue, Literal(">10%")))
        else:
            g.add((k, kpiName, Literal("satisfaction")))
            g.add((k, kpiValue, Literal(">80%")))
        g.add((p, hasKPI, k))

        # Dimensions
        for j in range(random.randint(1, 3)):
            dim = uri("Dim", i, j)
            g.add((dim, RDF.type, DXDimension))
            g.add((dim, dimensionDesc, Literal(random.choice(dimensions))))
            g.add((p, hasDimension, dim))

        # Phase
        g.add((p, isInPhase, Literal(random.choice(phases))))

        # Resources
        n_docs = random.randint(*docs_per_project)
        for j in range(n_docs):
            res = uri("Res", i, j)
            g.add((res, RDF.type, DxResourceType))
            ftype = random.choice(filetypes)
            fname = f"{pname}_{j}.{ 'jpg' if ftype=='image' else ('docx' if ftype=='word' else ftype) }"
            g.add((res, hasFileType, Literal(ftype)))
            g.add((res, hasFileName, Literal(fname)))
            g.add((p, hasResource, res))

    return g


# --------------------
# SPARQL queries
# --------------------
Q1_SIMPLE_PROJECT_FILES = """
PREFIX dx: <http://example.com/dx#>
SELECT ?projectName ?fileType ?fileName WHERE {
  ?p a dx:DXProject ;
     dx:projectName ?projectName ;
     dx:hasResource ?res .
  ?res a dx:DxResourceType ;
       dx:hasFileType ?fileType ;
       dx:hasFileName ?fileName .
  FILTER(STRSTARTS(?projectName, "Project_1"))
  FILTER(?fileType = "image")
} 
"""

Q2_CROSS_PROJECT_DRONE_SOUTH = """
PREFIX dx: <http://example.com/dx#>
SELECT ?projectName ?province WHERE {
  ?p a dx:DXProject ;
     dx:projectName ?projectName ;
     dx:hasProjectDetail ?detail ;
     dx:hasTechProduct ?tech ;
     dx:hasProjectResult ?result .
  ?detail dx:location ?province .
  ?tech dx:productName ?techProductName .
  ?result dx:isProjectResult "successful" .
  FILTER(CONTAINS(LCASE(?techProductName), "drone"))
  FILTER(?province = "Yala" || ?province = "Songkhla" || ?province = "Pattani" || ?province = "Satun")
}
"""

Q3_COUNT_SUCCESS_BY_TECH = """
PREFIX dx: <http://example.com/dx#>
SELECT ?tech ?count WHERE {
  {
    SELECT ?tech (COUNT(?p) AS ?count) WHERE {
      ?p a dx:DXProject ;
         dx:hasTechProduct ?t ;
         dx:hasProjectResult ?r .
      ?t dx:productName ?tech .
      ?r dx:isProjectResult "successful" .
    } GROUP BY ?tech
  }
} ORDER BY DESC(?count)
"""

QUERIES: List[Tuple[str, str]] = [
    ("Q1_simple_project_files", Q1_SIMPLE_PROJECT_FILES),
    ("Q2_cross_project_drone_south", Q2_CROSS_PROJECT_DRONE_SOUTH),
    ("Q3_count_success_by_tech", Q3_COUNT_SUCCESS_BY_TECH),
]


# --------------------
# Timing harness
# --------------------
def time_query(g: Graph, query: str, runs: int = 20, warmups: int = 3) -> Tuple[float, float]:
    """Run a SPARQL query multiple times; returns (median_ms, p95_ms)."""
    for _ in range(warmups):
        list(g.query(query))
    durs_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        list(g.query(query))
        durs_ms.append((time.perf_counter() - t0) * 1000.0)
    median = statistics.median(durs_ms)
    p95 = sorted(durs_ms)[int(0.95 * len(durs_ms)) - 1] if len(durs_ms) >= 20 else max(durs_ms)
    return round(median, 3), round(p95, 3)


# --------------------
# Aggregation
# --------------------
def aggregate_means(per_trial_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Average median_ms and p95_ms across trials for each (projects, query).
    Returns aggregated rows.
    """
    buckets: Dict[Tuple[int, str], List[Tuple[float, float]]] = defaultdict(list)
    meta_example: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for r in per_trial_rows:
        key = (int(r["projects"]), r["query"])
        buckets[key].append((float(r["median_ms"]), float(r["p95_ms"])))
        meta_example[key] = {
            "runs": r["runs"],
            "warmups": r["warmups"],
            "docs_min": r["docs_min"],
            "docs_max": r["docs_max"],
        }

    out: List[Dict[str, Any]] = []
    for (projects, query), vals in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        medians = [v[0] for v in vals]
        p95s = [v[1] for v in vals]
        meta = meta_example[(projects, query)]
        out.append({
            "projects": projects,
            "query": query,
            "trials": len(vals),
            "mean_median_ms": round(sum(medians) / len(medians), 3),
            "mean_p95_ms": round(sum(p95s) / len(p95s), 3),
            "runs": meta["runs"],
            "warmups": meta["warmups"],
            "docs_min": meta["docs_min"],
            "docs_max": meta["docs_max"],
        })
    return out


# --------------------
# CSV utilities
# --------------------
def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------
# Main: suite runner
# --------------------
def main():
    ap = argparse.ArgumentParser(description="DX-Onto SPARQL benchmark suite (rdflib)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500, 1000],
                    help="Project sizes to test")
    ap.add_argument("--trials", type=int, default=5, help="Trials per size")
    ap.add_argument("--runs", type=int, default=20, help="Timed runs per query")
    ap.add_argument("--warmups", type=int, default=3, help="Warmups per query")
    ap.add_argument("--docs-min", type=int, default=2, help="Min docs per project")
    ap.add_argument("--docs-max", type=int, default=5, help="Max docs per project")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--out", type=str, default="dxonto_suite_results.csv", help="Per-trial CSV path")
    ap.add_argument("--summary", type=str, default="dxonto_suite_summary.csv", help="Aggregated CSV path")
    args = ap.parse_args()

    per_trial_rows: List[Dict[str, Any]] = []

    print("=== DX-Onto SPARQL Benchmark Suite ===")
    print(f"Sizes: {args.sizes} | Trials per size: {args.trials} | Runs: {args.runs} | Warmups: {args.warmups}")
    print(f"Docs per project: [{args.docs_min}, {args.docs_max}] | Base seed: {args.seed}")
    print("--------------------------------------")

    for projects in args.sizes:
        for trial in range(1, args.trials + 1):
            seed = args.seed + trial  # vary seed per trial
            g = build_graph(projects, (args.docs_min, args.docs_max), seed)
            triples = len(g)

            print(f"[Size {projects:>4}, Trial {trial}] Graph triples: {triples}")
            for qname, qtext in QUERIES:
                median_ms, p95_ms = time_query(g, qtext, runs=args.runs, warmups=args.warmups)
                print(f"  {qname:<30} median={median_ms:7.2f} ms   p95={p95_ms:7.2f} ms")
                per_trial_rows.append({
                    "projects": projects,
                    "trial": trial,
                    "query": qname,
                    "median_ms": median_ms,
                    "p95_ms": p95_ms,
                    "runs": args.runs,
                    "warmups": args.warmups,
                    "triples": triples,
                    "docs_min": args.docs_min,
                    "docs_max": args.docs_max,
                    "seed": seed,
                })

    # Write per-trial CSV
    if args.out:
        write_csv(per_trial_rows, args.out)
        print(f"\nWrote per-trial results: {args.out}")

    # Aggregate and write summary CSV
    summary_rows = aggregate_means(per_trial_rows)
    if args.summary:
        write_csv(summary_rows, args.summary)
        print(f"Wrote aggregated summary: {args.summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()
