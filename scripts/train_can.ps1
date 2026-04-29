# 训练 can 类别的批处理脚本
$env:PYTHONPATH="src"

Write-Host "开始训练 can 类别..." -ForegroundColor Green
Write-Host "训练可能需要一些时间，请耐心等待..." -ForegroundColor Yellow

python bin/train_mvtec_ad2_single.py `
    --data_path ./mvtec_ad_2 `
    --classname can `
    --results_path ./results_can `
    --gpu 0 `
    --backbone wideresnet50 `
    --imagesize 224 `
    --save_model

if ($LASTEXITCODE -eq 0) {
    Write-Host "训练完成！模型保存在 ./results_can/models/can" -ForegroundColor Green
} else {
    Write-Host "训练失败，请检查错误信息" -ForegroundColor Red
}

