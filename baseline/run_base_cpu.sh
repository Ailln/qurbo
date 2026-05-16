python baseline_miqp_qaoa.py \
  --input ../data/alpha-test/miqp_sample_A.npz \
  --output solution_A_cpu.npz \
  --iterations 10 \
  --sub-size 12 \
  --shots 256 \
  --device CPU
