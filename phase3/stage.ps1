param([string]$Volume = '4q8k7ibb17')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$aws = Get-Command aws -ErrorAction SilentlyContinue
$cli = if ($aws) { $aws.Source } else { Join-Path $env:ProgramFiles 'Amazon\AWSCLIV2\aws.exe' }
$opt = @('--profile','signgpt-volume','--region','EUR-IS-1','--endpoint-url','https://s3api-eur-is-1.runpod.io/')
$env:AWS_REQUEST_CHECKSUM_CALCULATION = 'when_required'
$env:AWS_RESPONSE_CHECKSUM_VALIDATION = 'when_required'
$audit = Join-Path $PSScriptRoot 'audit-data'

function Put($from, $key) {
    & $cli s3 cp $from "s3://$Volume/$key" --only-show-errors @opt
    if ($LASTEXITCODE -ne 0) { throw "Staging failed: $key" }
}

$units = Get-Content -LiteralPath (Join-Path $audit 'pose-audit.json') -Raw | ConvertFrom-Json
if (@($units | Where-Object { -not $_.frame_counts_match }).Count -gt 0) {
    Write-Host 'The follow-up will exclude frame-count mismatches.'
}
foreach ($split in @('diagnostic','heldout')) {
    if (-not ($units | Where-Object { $_.frame_counts_match -and $_.split -eq $split })) {
        throw "No exact-frame pairs in $split"
    }
}
foreach ($file in @('benchmark.py','captions.py','pose_data.py','run_validation.py','score.py')) {
    Put (Join-Path $root "phase2/$file") "phase2/$file"
}
foreach ($file in @('followup.py','desktop.py','ui/index.html','ui/style.css','ui/app.js')) {
    Put (Join-Path $PSScriptRoot $file) "phase3/$file"
}
& $cli s3 cp $audit "s3://$Volume/signgpt-validation/prepared/phase3/" --recursive `
    --exclude '*' --include '*.npz' --include 'publisher.jsonl' --include 'fresh.jsonl' --only-show-errors @opt
if ($LASTEXITCODE -ne 0) { throw 'Staging failed: Phase 3 pose pairs' }
Put (Join-Path $audit 'pose-audit.json') 'signgpt-validation/prepared/phase3/pose-audit.json'
Write-Host 'Phase 3 exact-frame inputs and code staged. No raw video uploaded.'
