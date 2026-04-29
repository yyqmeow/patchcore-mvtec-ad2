@echo off
set PYTHONPATH=src
echo ========================================
echo 评估 can 类别 - 使用改进后的 Tiling（最大值合并策略）
echo ========================================
python bin/evaluate_mvtec_ad2_single.py --data_path ./mvtec_ad_2 --classname can --model_path ./results_can/models/can --results_path ./evaluation_results_can_tiling_improved --gpu 0 --imagesize 512 --use_tiling --tile_size 224 --tile_stride 112 --save_images

echo.
echo ========================================
echo 评估完成！
echo ========================================
echo 结果保存在: evaluation_results_can_tiling_improved/results_can.csv
pause

