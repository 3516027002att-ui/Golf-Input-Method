param(
    [string]$DataRoot = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$defaultDriveFolder = [string]::Concat([char]0x6211, [char]0x7684, [char]0x4E91, [char]0x7AEF, [char]0x786C, [char]0x76D8)
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path -Path ("G:\" + $defaultDriveFolder) -ChildPath "golf-ime-data-rebuild\clean_dataset_v3\left_context_only"
}

$trainJsonl = Join-Path $DataRoot "train_new_corpus.jsonl"
$valJsonl = Join-Path $DataRoot "val_new_corpus_v2.jsonl"
$testJsonl = Join-Path $DataRoot "test_new_corpus_v2.jsonl"
$reportPath = "reports\context_v2_new_corpus_audit.md"

foreach ($path in @($trainJsonl, $valJsonl, $testJsonl)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing context reranker v2 data file: $path"
    }
}

Push-Location $repoRoot
try {
    & python "training\context_reranker_v2.py" audit-splits `
        --train $trainJsonl `
        --val $valJsonl `
        --test $testJsonl `
        --report $reportPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
