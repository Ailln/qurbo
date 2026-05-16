python baseline_miqp_qaoa.py \
  --input miqp_sample_A.npz \
  --output solution_A_gpu.npz \
  --iterations 10 \
  --sub-size 12 \
  --shots 256 \
  --device GPU
