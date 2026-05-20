@echo off
cd /d "c:\Users\manue\Desktop\TourismRouteSystem"
python evals/run_benchmark.py ^
  --input evals/prompts_490_balanced.txt ^
  --timeout 700 ^
  --throttle 2.0 ^
  --resume ^
  --output-dir outputs/benchmark_20260520_093159 ^
  >> outputs/benchmark_490_run.log 2>&1
