param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cmin_us", "cmin_cn")]
    [string]$Market
)
$ErrorActionPreference = "Stop"
python train_base.py --market $Market
