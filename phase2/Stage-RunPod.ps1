param(
    [Parameter(Mandatory)][string]$Stage,
    [ValidateSet('Code','Models','Oracle','Continuous','All')][string]$Part = 'All',
    [Parameter(Mandatory)][string]$Volume,
    [string]$Region = 'EUR-IS-1',
    [string]$Profile = 'signgpt-volume',
    [ValidateRange(0,7200)][int]$WaitSeconds = 0,
    [int]$AfterProcess = 0
)

$ErrorActionPreference = 'Stop'
# Keep AWS CLI's optional multipart checksums off for the RunPod S3 endpoint.
$env:AWS_REQUEST_CHECKSUM_CALCULATION = 'when_required'
$env:AWS_RESPONSE_CHECKSUM_VALIDATION = 'when_required'
$env:AWS_MAX_ATTEMPTS = '10'
if ($Volume -notmatch '^[A-Za-z0-9-]+$' -or $Region -notmatch '^[A-Z0-9-]+$') { throw 'Invalid volume or region' }
$aws = Get-Command aws -ErrorAction SilentlyContinue
$cli = if ($aws) { $aws.Source } else { Join-Path $env:ProgramFiles 'Amazon\AWSCLIV2\aws.exe' }
$settings = @('--profile', $Profile, '--region', $Region, '--endpoint-url', "https://s3api-$($Region.ToLowerInvariant()).runpod.io/", '--cli-read-timeout', '180')
$root = "s3://$Volume"

function Publish-Directory($local, $remote, $marker = '') {
    if ($marker -and $WaitSeconds) {
        $deadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
        while (-not (Test-Path -LiteralPath (Join-Path $local $marker)) -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds 15
        }
    }
    if (-not (Test-Path -LiteralPath $local -PathType Container)) { throw "Missing directory: $local" }
    if ($marker -and -not (Test-Path -LiteralPath (Join-Path $local $marker))) { throw "Missing completion file: $marker" }
    $excludes = @('--exclude','*.part','--exclude','*/__pycache__/*')
    if ($remote -eq 'phase2') {
        $excludes = @('--exclude','*','--include','*.py','--include','*.sh','--include','*.ps1',
                      '--include','*.patch','--include','*.md','--include','requirements.txt',
                      '--exclude','*/__pycache__/*')
    }
    if ($marker) { $excludes += @('--exclude',$marker) }
    if ($remote -eq 'signgpt-validation/assets') {
        & $cli s3 sync $local "$root/$remote/" --size-only --only-show-errors @excludes @settings
    } else {
        & $cli s3 cp $local "$root/$remote/" --recursive --only-show-errors @excludes @settings
    }
    if ($LASTEXITCODE -ne 0) { throw "Upload failed: $remote" }
    if ($marker) {
        if ($remote -eq 'signgpt-validation/assets') {
            & python -B (Join-Path $PSScriptRoot 'verify_model_upload.py') --aws $cli --volume $Volume --region $Region --profile $Profile $local (Join-Path $local $marker)
        } else {
            & $cli s3 cp (Join-Path $local $marker) "$root/$remote/$marker" --only-show-errors @settings
        }
        if ($LASTEXITCODE -ne 0) { throw "Completion file upload failed: $remote" }
    }
    "Staged $remote"
}

if ($Part -in @('Code','All')) { Publish-Directory $PSScriptRoot 'phase2' }
if ($Part -in @('Oracle','All')) { Publish-Directory (Join-Path $Stage 'prepared\how2sign-oracle') 'signgpt-validation/prepared/how2sign-oracle' 'oracle.jsonl' }
if ($Part -in @('Continuous','All')) { Publish-Directory (Join-Path $Stage 'continuous\prepared') 'signgpt-validation/prepared/continuous' 'continuous.jsonl' }
if ($Part -in @('Models','All')) {
    if ($AfterProcess) {
        Wait-Process -Id $AfterProcess -ErrorAction SilentlyContinue
        $pending = Join-Path $Stage 'pending-models.sha256'
        $completed = Join-Path $Stage 'assets\staged.sha256'
        if ((Test-Path -LiteralPath $pending) -and -not (Test-Path -LiteralPath $completed)) {
            Move-Item -LiteralPath $pending -Destination $completed
        }
    }
    # Isolate transfer settings while continuing to use the existing profile credentials.
    $oldConfig = $env:AWS_CONFIG_FILE
    $config = Join-Path $Stage ("s3-model-transfer-{0}.config" -f $PID)
    @"
[profile $Profile]
region = $Region
s3 =
    preferred_transfer_client = classic
    max_concurrent_requests = 2
    multipart_chunksize = 5MB
"@ | Set-Content -LiteralPath $config -Encoding ASCII
    $env:AWS_CONFIG_FILE = $config
    try {
        Publish-Directory (Join-Path $Stage 'assets') 'signgpt-validation/assets' 'staged.sha256'
    } finally {
        $env:AWS_CONFIG_FILE = $oldConfig
        Remove-Item -LiteralPath $config -ErrorAction SilentlyContinue
    }
}
