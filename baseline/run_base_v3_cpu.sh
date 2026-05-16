python baseline_miqp_qaoa_v3.py \
  --input ../data/alpha-test/miqp_sample_B.npz \
  --output solution_B_v3_cpu.npz \
  --iterations 120 \
  --sub-size 20 \
  --shots 512 \
  --top-k 30 \
  --candidate-pool 30 \
  --device CPU
