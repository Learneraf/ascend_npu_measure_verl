第一部分的实验
cd /data/yanziyi/gpu_test_0610
bash run_qwen3_8b_a100.sh
bash run_qwen3_32b_lora_a100.sh
第二部分的实验
cd /data/yanziyi/gpu_test_0610/ablation
bash ablation_rollout_n.sh
bash ablation_rollout_n_32b.sh
bash ablation_batchsize.sh
bash ablation_batchsize_32b.sh
bash ablation_seqlen.sh
bash ablation_seqlen_32b.sh
第三部分的实验
cd /data/yanziyi/gpu_test_0610/other_scripts
bash profile_rollout.sh
bash profile_rollout_32b.sh
第四部分的实验（建议verl版本0.8.x及以上）
cd /data/yanziyi/gpu_test_0610/other_scripts
bash run_async.sh
bash run_async_32b.sh

运行完毕后结果自动填入results_tables.md