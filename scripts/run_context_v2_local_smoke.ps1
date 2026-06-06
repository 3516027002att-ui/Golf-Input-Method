[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [string]$PythonExe = "python",
    [string]$Encoder = "hf-internal-testing/tiny-random-bert",
    [string]$Device = "cpu",
    [int]$TrainRows = 64,
    [int]$ValRows = 24,
    [int]$TestRows = 24,
    [int]$Epochs = 1,
    [int]$BatchSize = 2,
    [int]$EvalBatchSize = 4,
    [switch]$FailOnSanityRedLine
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$defaultDriveFolder = [string]::Concat([char]0x6211, [char]0x7684, [char]0x4E91, [char]0x7AEF, [char]0x786C, [char]0x76D8)
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path -Path ("G:\" + $defaultDriveFolder) -ChildPath "golf-ime-data-rebuild\clean_dataset_v3\left_context_only"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$smokeRoot = Join-Path $repoRoot ".smoke_data\context_v2_new_corpus\$runId"
$runRoot = Join-Path $repoRoot "reports\context_v2_local_smoke\$runId"
$checkpointRoot = Join-Path $runRoot "checkpoints"
$logRoot = Join-Path $runRoot "logs"

$trainJsonl = Join-Path $DataRoot "train_new_corpus.jsonl"
$valJsonl = Join-Path $DataRoot "val_new_corpus_v2.jsonl"
$testJsonl = Join-Path $DataRoot "test_new_corpus_v2.jsonl"

function Assert-FileExists([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing required file: $Path"
    }
}

function New-Directory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Copy-JsonlHead {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][int]$Limit
    )

    New-Directory (Split-Path -Parent $Destination)
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $reader = [System.IO.StreamReader]::new($Source, $utf8NoBom, $true)
    $writer = [System.IO.StreamWriter]::new($Destination, $false, $utf8NoBom)
    try {
        $written = 0
        while (-not $reader.EndOfStream -and $written -lt $Limit) {
            $line = $reader.ReadLine()
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $writer.WriteLine($line.TrimEnd())
                $written += 1
            }
        }
    }
    finally {
        $writer.Dispose()
        $reader.Dispose()
    }
    if ($written -lt 1) {
        throw "No non-empty JSONL rows copied from $Source"
    }
    return $written
}

function Invoke-LoggedPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    New-Directory (Split-Path -Parent $LogPath)
    Write-Host ""
    Write-Host ("$ {0} {1}" -f $PythonExe, ($Arguments -join " "))
    $output = & $PythonExe @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    [System.IO.File]::WriteAllText($LogPath, $text + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    if ($text) {
        Write-Host $text
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $LogPath"
    }
    return $text
}

function Read-JsonObjectFromText {
    param([Parameter(Mandatory = $true)][string]$Path)
    $text = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    $start = $text.IndexOf("{")
    $end = $text.LastIndexOf("}")
    if ($start -lt 0 -or $end -lt $start) {
        throw "No JSON object found in $Path"
    }
    return ($text.Substring($start, $end - $start + 1) | ConvertFrom-Json)
}

function Get-JsonValue {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string[]]$Path
    )
    $cursor = $Object
    foreach ($part in $Path) {
        if ($null -eq $cursor) {
            return $null
        }
        $cursor = $cursor.$part
    }
    return $cursor
}

foreach ($path in @($trainJsonl, $valJsonl, $testJsonl)) {
    Assert-FileExists $path
}

New-Directory $smokeRoot
New-Directory $runRoot
New-Directory $checkpointRoot
New-Directory $logRoot

$smokeTrain = Join-Path $smokeRoot "train_smoke.jsonl"
$smokeVal = Join-Path $smokeRoot "val_smoke.jsonl"
$smokeTest = Join-Path $smokeRoot "test_smoke.jsonl"

$actualTrainRows = Copy-JsonlHead -Source $trainJsonl -Destination $smokeTrain -Limit $TrainRows
$actualValRows = Copy-JsonlHead -Source $valJsonl -Destination $smokeVal -Limit $ValRows
$actualTestRows = Copy-JsonlHead -Source $testJsonl -Destination $smokeTest -Limit $TestRows

$auditReport = Join-Path $runRoot "smoke_audit.md"
Invoke-LoggedPython -Arguments @(
    "training\context_reranker_v2.py", "audit-splits",
    "--train", $smokeTrain,
    "--val", $smokeVal,
    "--test", $smokeTest,
    "--report", $auditReport
) -LogPath (Join-Path $logRoot "audit.log") | Out-Null

$onlineCheckpoint = Join-Path $checkpointRoot "online"
$randomCheckpoint = Join-Path $checkpointRoot "random_label"

$commonTrainArgs = @(
    "training\context_reranker_v2.py", "train",
    "--train", $smokeTrain,
    "--val", $smokeVal,
    "--encoder", $Encoder,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--eval-batch-size", "$EvalBatchSize",
    "--max-length", "64",
    "--context-before-chars", "48",
    "--context-after-chars", "24",
    "--device", $Device,
    "--log-every", "1"
)

Invoke-LoggedPython -Arguments ($commonTrainArgs + @(
    "--output-dir", $onlineCheckpoint,
    "--context-mode", "online"
)) -LogPath (Join-Path $logRoot "train_online.log") | Out-Null

Invoke-LoggedPython -Arguments ($commonTrainArgs + @(
    "--output-dir", $randomCheckpoint,
    "--context-mode", "online",
    "--label-mode", "random"
)) -LogPath (Join-Path $logRoot "train_random_label.log") | Out-Null

$onlineEvalLog = Join-Path $runRoot "online_eval.json"
$noContextEvalLog = Join-Path $runRoot "online_checkpoint_no_context_eval.json"
$shuffleEvalLog = Join-Path $runRoot "online_checkpoint_shuffle_eval.json"
$randomEvalLog = Join-Path $runRoot "random_label_eval.json"

Invoke-LoggedPython -Arguments @(
    "training\context_reranker_v2.py", "eval",
    "--data", $smokeTest,
    "--checkpoint", $onlineCheckpoint,
    "--batch-size", "$EvalBatchSize",
    "--context-mode", "online",
    "--device", $Device
) -LogPath $onlineEvalLog | Out-Null

Invoke-LoggedPython -Arguments @(
    "training\context_reranker_v2.py", "eval",
    "--data", $smokeTest,
    "--checkpoint", $onlineCheckpoint,
    "--batch-size", "$EvalBatchSize",
    "--context-mode", "none",
    "--device", $Device
) -LogPath $noContextEvalLog | Out-Null

Invoke-LoggedPython -Arguments @(
    "training\context_reranker_v2.py", "eval",
    "--data", $smokeTest,
    "--checkpoint", $onlineCheckpoint,
    "--batch-size", "$EvalBatchSize",
    "--context-mode", "online",
    "--candidate-order", "shuffle",
    "--device", $Device
) -LogPath $shuffleEvalLog | Out-Null

Invoke-LoggedPython -Arguments @(
    "training\context_reranker_v2.py", "eval",
    "--data", $smokeTest,
    "--checkpoint", $randomCheckpoint,
    "--batch-size", "$EvalBatchSize",
    "--context-mode", "online",
    "--device", $Device
) -LogPath $randomEvalLog | Out-Null

$predictLog = Join-Path $runRoot "predict_xuexiao.json"
$schoolCandidate = [string]::Concat([char]0x5B66, [char]0x6821)
$sleepCandidate = [string]::Concat([char]0x7761, [char]0x89C9)
$beijingCandidate = [string]::Concat([char]0x5317, [char]0x4EAC)
$oneCandidate = [string]::Concat([char]0x4E00, [char]0x4E2A)
$predictContextBefore = [string]::Concat([char]0x6211, [char]0x4ECA, [char]0x5929, [char]0x60F3, [char]0x53BB)
$candidatesJson = ConvertTo-Json -Compress -InputObject @(
    $schoolCandidate,
    $sleepCandidate,
    $beijingCandidate,
    $oneCandidate
)

$runnerPath = Join-Path $runRoot "predict_runner.py"
$runnerCode = @(
    "import sys"
    "import json"
    "sys.path.insert(0, r'$repoRoot')"
    "from scripts.predict_context_reranker_v2 import run_prediction, build_parser"
    "parser = build_parser()"
    "args = parser.parse_args(["
    "    '--checkpoint', r'$onlineCheckpoint',"
    "    '--context-before', '$predictContextBefore',"
    "    '--composing', 'xuexiao',"
    "    '--candidates-json', '$candidatesJson',"
    "    '--context-mode', 'online',"
    "    '--device', '$Device'"
    "])"
    "result = run_prediction(args)"
    "print(json.dumps(result, ensure_ascii=True, indent=2))"
) -join [Environment]::NewLine
[System.IO.File]::WriteAllText($runnerPath, $runnerCode, [System.Text.UTF8Encoding]::new($false))

Invoke-LoggedPython -Arguments @($runnerPath) -LogPath $predictLog | Out-Null

$onlineEval = Read-JsonObjectFromText $onlineEvalLog
$noContextEval = Read-JsonObjectFromText $noContextEvalLog
$shuffleEval = Read-JsonObjectFromText $shuffleEvalLog
$randomEval = Read-JsonObjectFromText $randomEvalLog
$predict = Read-JsonObjectFromText $predictLog

$onlineTop1 = [double](Get-JsonValue $onlineEval @("metrics", "top1"))
$noContextTop1 = [double](Get-JsonValue $noContextEval @("metrics", "top1"))
$shuffleTop1 = [double](Get-JsonValue $shuffleEval @("metrics", "top1"))
$randomLabelTop1 = [double](Get-JsonValue $randomEval @("metrics", "top1"))
$randomBaselineTop1 = [double](Get-JsonValue $randomEval @("random_baseline", "top1"))
$rankedCandidates = @($predict.ranked_candidates)
$schoolRank = $null
for ($index = 0; $index -lt $rankedCandidates.Count; $index += 1) {
    if ($rankedCandidates[$index].candidate -eq $schoolCandidate) {
        $schoolRank = $index + 1
        break
    }
}
$firstCandidate = if ($rankedCandidates.Count -gt 0) { $rankedCandidates[0].candidate } else { $null }

$checks = [ordered]@{
    online_beats_no_context = ($onlineTop1 -gt $noContextTop1)
    online_minus_no_context = ($onlineTop1 - $noContextTop1)
    shuffle_not_better_than_online = ($shuffleTop1 -le ($onlineTop1 + 0.000001))
    random_label_not_better_than_online = ($randomLabelTop1 -le ($onlineTop1 + 0.000001))
    random_label_near_random_baseline = ([Math]::Abs($randomLabelTop1 - $randomBaselineTop1) -le 0.25)
    predict_school_first = ($firstCandidate -eq $schoolCandidate)
}

$summary = [ordered]@{
    run_id = $runId
    run_dir = "$runRoot"
    smoke_data_dir = "$smokeRoot"
    data_files = [ordered]@{
        train = $trainJsonl
        val = $valJsonl
        test = $testJsonl
        excluded_first_round = Join-Path $DataRoot "train.jsonl"
    }
    sample_rows = [ordered]@{
        train = $actualTrainRows
        val = $actualValRows
        test = $actualTestRows
    }
    encoder = $Encoder
    device = $Device
    checkpoints = [ordered]@{
        online = $onlineCheckpoint
        random_label = $randomCheckpoint
    }
    outputs = [ordered]@{
        audit_report = $auditReport
        online_eval = $onlineEvalLog
        no_context_eval = $noContextEvalLog
        shuffle_eval = $shuffleEvalLog
        random_label_eval = $randomEvalLog
        predict_xuexiao = $predictLog
    }
    metrics = [ordered]@{
        online_top1 = $onlineTop1
        no_context_top1 = $noContextTop1
        shuffle_top1 = $shuffleTop1
        random_label_top1 = $randomLabelTop1
        random_baseline_top1 = $randomBaselineTop1
    }
    checks = $checks
    eval_red_line_findings = [ordered]@{
        online = @($onlineEval.red_line_findings)
        no_context = @($noContextEval.red_line_findings)
        shuffle = @($shuffleEval.red_line_findings)
        random_label = @($randomEval.red_line_findings)
    }
    prediction = [ordered]@{
        first_candidate = $firstCandidate
        school_rank = $schoolRank
        ranked_candidates = $rankedCandidates
    }
}

$summaryPath = Join-Path $runRoot "summary.json"
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

foreach ($entry in $checks.GetEnumerator()) {
    if (-not $entry.Value -and $entry.Key -ne "predict_school_first") {
        Write-Warning "Sanity check did not pass on smoke data: $($entry.Key)"
    }
}
if (-not $checks.predict_school_first) {
    Write-Warning "Predict smoke did not rank the xuexiao school candidate first. This tiny checkpoint is only a chain test; rerun with the formal Colab checkpoint for product validation."
}
$redLines = @($summary.eval_red_line_findings.online + $summary.eval_red_line_findings.no_context + $summary.eval_red_line_findings.shuffle + $summary.eval_red_line_findings.random_label) | Where-Object { $_ }
if ($redLines.Count -gt 0) {
    Write-Warning "Eval red lines were reported. See summary.json; they are not silently ignored."
}
if ($FailOnSanityRedLine -and (($checks.Values -contains $false) -or $redLines.Count -gt 0)) {
    throw "Sanity red line detected. See $summaryPath"
}

Write-Host ""
Write-Host "CONTEXT_V2_LOCAL_SMOKE_DONE"
Write-Host "run_dir=$runRoot"
Write-Host "summary=$summaryPath"
Write-Host "audit_report=$auditReport"
Write-Host "online_checkpoint=$onlineCheckpoint"
