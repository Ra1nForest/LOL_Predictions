# 在 Windows 上部署四个任务, 对应服务器的四个 systemd 单元。
#
# 为什么会有这个脚本: 2026-08-30 本机 WSL2 因 8/12 的 Windows 更新损坏了
# 四个虚拟化 ROOT 设备节点 (ROOT\VDRVROOT / HVSERVICE / UMBUS /
# NDISVIRTUALBUS 全部 Error), 导致 Windows 挂不上任何 VHD —— WSL2 无论哪个
# 版本都起不来。同时 Oracle 那台 E5.Flex 已随试用到期降级, 配额归零、重建不
# 能。所以生产环境先落到 Windows 原生, 等 WSL 修好再迁回去。
#
# 与 systemd 版的关键差异 —— 照搬会静默出事:
#
#   1. **计划任务默认不补跑。** systemd 的 Persistent=true 在这里对应
#      -StartWhenAvailable; 不加的话关机期间错过的 update 直接消失, 不报错。
#   2. **时区。** 服务器的 OnCalendar 用 UTC, 计划任务只认本地时间。本脚本
#      按本机时区把 UTC 时刻换算过去, 并把换算结果打出来供核对。
#   3. **daily_update 的重启动作。** 它原来调 `sudo systemctl restart`,
#      在 Windows 上必然失败。daily_update.py 已改成平台感知 (schtasks
#      /end + /run), 这里的任务名必须和传给它的 --service 一致。
#   4. **不存凭据。** 任务以当前用户、仅在登录时运行 —— 不用 -RunLevel
#      Highest 也不存密码。代价是必须保持登录状态 (锁屏可以)。
#
# 用法 (普通 PowerShell 即可, 会按需弹 UAC):
#     powershell -ExecutionPolicy Bypass -File deploy\windows_install.ps1
#     ... -Remove      卸载全部任务

param([switch]$Remove)

$ErrorActionPreference = 'Stop'

# 调用原生 exe (python/pip) 时必须临时放宽。PowerShell 5.1 把原生命令写到
# stderr 的**每一行**都包成 ErrorRecord, 配合 'Stop' 会让"测试全过"变成
# 脚本终止 —— unittest 的进度点恰恰走 stderr。用退出码判成败, 不看 stderr。
function Invoke-Native {
  param([scriptblock]$Block)
  $old = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Block } finally { $ErrorActionPreference = $old }
}
$Proj = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Proj '.venv'
$Py   = Join-Path $Venv 'Scripts\python.exe'
$Pyw  = Join-Path $Venv 'Scripts\pythonw.exe'

function Say($s){ Write-Host "`n== $s ==" -ForegroundColor Cyan }
function OK($s){ Write-Host "   [OK] $s" -ForegroundColor Green }
function Bad($s){ Write-Host "   [!!] $s" -ForegroundColor Red }

$Tasks = @('LoL-Predict','LoL-Collect','LoL-Update','LoL-Backfill')

if($Remove){
  Say '卸载'
  foreach($t in $Tasks){
    if(Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue){
      Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
      Unregister-ScheduledTask -TaskName $t -Confirm:$false
      OK "已删除 $t"
    }
  }
  return
}

Say '0. 环境'
Write-Host "   项目: $Proj"
if(-not (Test-Path (Join-Path $Proj 'api.py'))){ Bad '找不到 api.py, 路径不对'; exit 1 }
$sysPy = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if(-not $sysPy){ Bad '找不到 python.exe'; exit 1 }
OK "系统 Python: $(& python.exe --version 2>&1)"

Say '1. venv'
# 用独立 venv 而不是全局环境: 全局装的包会被别的项目升级, 生产服务不该跟着抖。
if(-not (Test-Path $Py)){
  Invoke-Native { & python.exe -m venv $Venv }
  OK '已创建 .venv'
} else { OK '.venv 已存在' }
Invoke-Native { & $Py -m pip install -q --upgrade pip }
Invoke-Native { & $Py -m pip install -q -r (Join-Path $Proj 'requirements.txt') }
OK "$(& $Py --version 2>&1)   $((& $Py -m pip list 2>$null | Measure-Object).Count) 个包"

Say '2. 回归测试 (装任务之前先确认代码能跑)'
Push-Location $Proj
$env:PYTHONIOENCODING = 'utf-8'
# **不要在这里写 2>&1**。PowerShell 5.1 会把原生命令的 stderr 每一行包成
# ErrorRecord (NativeCommandError) 并把 $? 置为 false —— 而 unittest 的进度点
# 恰恰走 stderr, 于是全部测试通过也会被判成失败。看退出码才是对的。
Invoke-Native { & $Py -m unittest discover -s tests }
$testRc = $LASTEXITCODE
Pop-Location
if($testRc -ne 0){ Bad "测试没过 (退出码 $testRc), 停止"; exit 1 }
OK '测试通过'

Say '3. 时区换算 (服务器用 UTC, 计划任务用本地时间)'
$tz = [TimeZoneInfo]::Local
Write-Host "   本机时区: $($tz.DisplayName)"
function UtcToLocalStr([int]$h,[int]$m){
  $utc = [DateTime]::SpecifyKind((Get-Date -Hour $h -Minute $m -Second 0), [DateTimeKind]::Utc)
  ([TimeZoneInfo]::ConvertTimeFromUtc($utc, $tz)).ToString('HH:mm')
}
$u00 = UtcToLocalStr 0 0     # lol-update 第一次
$u12 = UtcToLocalStr 12 0    # lol-update 第二次
$b13 = UtcToLocalStr 13 30   # lol-backfill
Write-Host "   lol-update  00:00 UTC -> $u00 本地"
Write-Host "   lol-update  12:00 UTC -> $u12 本地"
Write-Host "   lol-backfill 13:30 UTC -> $b13 本地"

Say '4. 注册任务'
# 全部用 pythonw.exe (无控制台) 启动, 日志由 Python 自己重定向:
#   LoL-Predict -> run_api.py 内部重定向
#   其余三个    -> run_task.py 统一启动器
# **不能包 cmd.exe 做重定向**: cmd 自己就是个窗口, 计划任务的 -Hidden 压不住,
# 每 2 分钟的 collect 会一直闪窗。2026-08-30 实测踩过。
$logDir = Join-Path $Proj 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
foreach($t in $Tasks){
  if(Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue){
    Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $t -Confirm:$false
  }
}

# 公共设置。StartWhenAvailable 是 systemd Persistent=true 的对应物 ——
# 没有它, 关机/休眠期间错过的触发会**静默消失**。
$common = @{
  AllowStartIfOnBatteries    = $true
  DontStopIfGoingOnBatteries = $true
  StartWhenAvailable         = $true
  ExecutionTimeLimit         = (New-TimeSpan -Hours 4)
  MultipleInstances          = 'IgnoreNew'
}

# ── LoL-Predict: 常驻 API ──
# pythonw + run_api.py: pythonw 保证没有控制台窗口, 而 run_api.py 在 import
# uvicorn **之前**就把 sys.stdout/stderr 接到日志文件上 —— 少了这一步,
# pythonw 下 stdout 是 None, uvicorn 第一行日志就让它崩, 退出码 1 且无处
# 可查。
# **不能包 cmd.exe 做重定向**: schtasks /end 只杀 cmd 外壳, python 被孤儿化
# 后继续占着 8000 端口跑旧模型 —— 于是 daily_update 换完 artifacts "重启成功"
# 、探活也过, 应答的却是那个旧进程。2026-08-30 实测踩过。
# 单 worker: FeatureStore 常驻约 1.3GB, 多开纯属浪费。
$a = New-ScheduledTaskAction -Execute $Pyw -Argument 'run_api.py' -WorkingDirectory $Proj
$s = New-ScheduledTaskSettingsSet @common -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$s.ExecutionTimeLimit = 'PT0S'      # 常驻任务不能有超时, 否则 4 小时后被杀
Register-ScheduledTask -TaskName 'LoL-Predict' -Action $a `
  -Trigger (New-ScheduledTaskTrigger -AtLogOn) -Settings $s `
  -Description 'LoL 预测 API (常驻, 127.0.0.1:8000)' | Out-Null
OK 'LoL-Predict  (登录时启动, 失败重试 5 次)'

# ── LoL-Collect: 每 2 分钟采一次实时快照 ──
# 不设 StartWhenAvailable 的补跑意义 —— 实时采集没有"补跑"这回事, 错过就
# 是永久空洞。但保留它没有坏处。
$a = New-ScheduledTaskAction -Execute $Pyw -Argument 'run_task.py collect collect_live.py' -WorkingDirectory $Proj
$trig = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(3) `
          -RepetitionInterval (New-TimeSpan -Minutes 2)
$s = New-ScheduledTaskSettingsSet @common -Hidden
$s.ExecutionTimeLimit = 'PT10M'
Register-ScheduledTask -TaskName 'LoL-Collect' -Action $a -Trigger $trig -Settings $s `
  -Description '每 2 分钟采样进行中的比赛 (局内训练数据)' | Out-Null
OK 'LoL-Collect  (每 2 分钟)'

# ── LoL-Update: 每天两次, 拉数据 + 重训 + 过闸 + 换 artifacts + 重启 ──
# --service 必须是 LoL-Predict: daily_update.py 在 Windows 上用
# schtasks /end + /run 重启它。名字对不上就会走"没这个任务"分支, 结果是
# 新模型在磁盘、服务还在用旧的 —— 和服务器上坏了十天的那个 bug 一模一样。
$a = New-ScheduledTaskAction -Execute $Pyw -Argument 'run_task.py update daily_update.py --service LoL-Predict' -WorkingDirectory $Proj
$s = New-ScheduledTaskSettingsSet @common -Hidden
Register-ScheduledTask -TaskName 'LoL-Update' -Action $a `
  -Trigger @((New-ScheduledTaskTrigger -Daily -At $u00), (New-ScheduledTaskTrigger -Daily -At $u12)) `
  -Settings $s -Description "每天两次: 拉 OE 数据 + 重训 + 过闸 + 换模型 ($u00 / $u12)" | Out-Null
OK "LoL-Update   ($u00 / $u12, 错过会补跑)"

# ── LoL-Backfill: 每天一次, 必须在 update 之后 ──
# 它拿 Oracle's Elixir 的 result 当标签, OE 得先拉下来。顺序反了会静默跳过
# 当天所有比赛。
$a = New-ScheduledTaskAction -Execute $Pyw -Argument 'run_task.py backfill research\backfill_late.py --days 95' -WorkingDirectory $Proj
$s = New-ScheduledTaskSettingsSet @common
Register-ScheduledTask -TaskName 'LoL-Backfill' -Action $a `
  -Trigger (New-ScheduledTaskTrigger -Daily -At $b13) -Settings $s `
  -Description "每天一次回填已结束比赛的分钟级快照 ($b13, 必须在 Update 之后)" | Out-Null
OK "LoL-Backfill ($b13, 错过会补跑)"

Say '5. 启动 API 并探活'
Start-ScheduledTask -TaskName 'LoL-Predict'
$ready = $false
for($i=1; $i -le 40; $i++){
  Start-Sleep -Seconds 5
  try{
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3 -UseBasicParsing
    if($r.StatusCode -eq 200){ OK "/health 就绪 (约 $($i*5) 秒)"; $ready = $true; break }
  }catch{}
}
if(-not $ready){ Bad '探活超时, 看任务历史: Get-ScheduledTaskInfo LoL-Predict' }
else {
  $h = (Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing).Content | ConvertFrom-Json
  foreach($k in $h.stages.PSObject.Properties.Name){
    Write-Host ("   {0,-12} accuracy={1:N6}" -f $k, $h.stages.$k.accuracy)
  }
}

Say '完成'
Get-ScheduledTask -TaskName $Tasks | Select-Object TaskName,State | Format-Table -AutoSize
Write-Host @"
   UI:      http://127.0.0.1:8000/
   查状态:  Get-ScheduledTask LoL-*
   看历史:  Get-ScheduledTaskInfo LoL-Update
   卸载:    powershell -File deploy\windows_install.ps1 -Remove

   注意: 任务设为"仅在登录时运行", 所以要保持登录状态 (锁屏可以, 注销不行)。
"@
