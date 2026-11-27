$ErrorActionPreference = 'Stop'
$cli = Join-Path $env:ProgramFiles 'Amazon\AWSCLIV2\aws.exe'
$volume = '4q8k7ibb17'
$options = @('--profile','signgpt-volume','--region','EUR-IS-1','--endpoint-url','https://s3api-eur-is-1.runpod.io/','--cli-connect-timeout','10','--cli-read-timeout','90')
$env:AWS_REQUEST_CHECKSUM_CALCULATION = 'when_required'
$env:AWS_RESPONSE_CHECKSUM_VALIDATION = 'when_required'
$cache = Join-Path $env:LOCALAPPDATA 'Temp\signgpt-phase2-stage\ort-cache\rtmlib\hub\checkpoints'
foreach ($name in @('yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx','rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx')) {
    $file = Get-Item -LiteralPath (Join-Path $cache $name)
    $remote = "signgpt-validation/cache/rtmlib/hub/checkpoints/$name"
    $reply = & $cli s3api list-objects-v2 --bucket $volume --prefix $remote --output json @options
    if ($LASTEXITCODE -ne 0) { throw 'Volume lookup failed' }
    $matches = ($reply | ConvertFrom-Json).Contents | Where-Object { $_.Key -eq $remote -and $_.Size -eq $file.Length }
    if ($matches) { continue }
    if ($file.Length -gt 32MB) {
        & python -B (Join-Path $PSScriptRoot 'stage_pose.py') $cli $file.FullName $remote $volume
    } else {
        & $cli s3api put-object --bucket $volume --key $remote --body $file.FullName @options | Out-Null
    }
    if ($LASTEXITCODE -ne 0) { throw "Upload failed for $name" }
}
foreach ($name in @('desktop.py','live_guard.py','live_worker.py','install-cuda-pose.sh','requirements-cuda-benchmark.txt','warmup.jpg')) {
    & $cli s3 cp (Join-Path $PSScriptRoot $name) "s3://$volume/phase3/$name" --only-show-errors @options
    if ($LASTEXITCODE -ne 0) { throw "Upload failed for $name" }
}
'Live worker staged on the existing volume.'
