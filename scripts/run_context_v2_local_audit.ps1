$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$trainJsonl = "G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only\train_new_corpus.jsonl"
$valJsonl = "G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only\val_new_corpus_v2.jsonl"
$testJsonl = "G:\我的云端硬盘\golf-ime-data-rebuild\clean_dataset_v3\left_context_only\test_new_corpus_v2.jsonl"
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
