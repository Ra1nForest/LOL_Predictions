#!/usr/bin/env bash
# 把整套服务从 Oracle_AMD6X 迁到本机 WSL2。
#
# 与服务器版 install.sh 的关键差别 —— 这些不是风格问题, 照搬会出事:
#
#   1. **数据必须落在 ext4, 不能留在 /mnt/c。** 跨 9p 文件系统读五年 CSV
#      (387MB) 会把冷启动从一分钟拖到十几分钟。所以整个项目复制进
#      $HOME/lol, 而不是直接挂 Windows 目录跑。
#   2. **systemd 在 WSL 默认是关的。** 要 /etc/wsl.conf 里显式打开, 且必须
#      wsl --shutdown 之后才生效。
#   3. **WSL 不随 Windows 开机启动。** 得在 Windows 侧挂一个登录触发的计划
#      任务, 否则重启之后四个单元一个都不会跑, 而且不报错。
#   4. **电脑睡眠时 WSL 整个挂起。** lol-collect 每 2 分钟采一次实时快照,
#      睡眠期间的比赛就是永久空洞 —— 这是相对 7x24 服务器的真实退化, 无法
#      靠 Persistent=true 补救 (实时采集没有"补跑"这回事)。
#   5. 内存上限下调: WSL 被 .wslconfig 限到 24GB, 服务器那套 12G/20G 会把
#      整个 WSL 逼到边缘。
#
# 用法 (在 WSL 里跑):
#     LOL_SERVER=ubuntu@<服务器IP> LOL_SSH_KEY=/mnt/c/<密钥路径> \
#         bash /mnt/c/<项目目录>/deploy/wsl_migrate.sh
#     bash /mnt/c/<项目目录>/deploy/wsl_migrate.sh --skip-pull   # 不从服务器拉, 用本地已同步的副本
set -euo pipefail

# 路径和服务器地址不写死 (仓库是公开的): 项目目录按脚本自己的位置推,
# 服务器和密钥从环境变量取 —— 只有要从服务器拉的时候才需要
WIN_PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIN_KEY="${LOL_SSH_KEY:-}"
SRV="${LOL_SERVER:-}"
ROOT="$HOME/lol"
SVC="$ROOT/service"
VENV="$ROOT/venv312"
SKIP_PULL=0
[[ "${1:-}" == "--skip-pull" ]] && SKIP_PULL=1

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok()  { printf '   \033[32m✓\033[0m %s\n' "$*"; }
bad() { printf '   \033[31m✗\033[0m %s\n' "$*"; }

say "0. 环境检查"
U=$(whoami)
[[ "$U" == "root" ]] && { bad "别用 root 跑这个脚本"; exit 1; }
ok "用户 $U   home $HOME"
. /etc/os-release; ok "$PRETTY_NAME  kernel $(uname -r)"
if [[ -d /run/systemd/system ]]; then
  ok "systemd 已启用 (PID1=$(ps -p1 -o comm=))"
  HAVE_SYSTEMD=1
else
  bad "systemd 未启用 —— 先写 /etc/wsl.conf, 然后 wsl --shutdown 再跑一遍"
  HAVE_SYSTEMD=0
fi
AVAIL=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
ok "根分区可用 ${AVAIL}G"
(( AVAIL < 15 )) && { bad "至少要 15G (数据 466M + venv + 训练临时占用)"; exit 1; }

say "1. 打开 systemd (幂等)"
if [[ ! -f /etc/wsl.conf ]] || ! grep -q '^systemd=true' /etc/wsl.conf 2>/dev/null; then
  sudo tee /etc/wsl.conf >/dev/null <<EOF
[boot]
systemd=true

[interop]
# 保留调用 Windows 程序的能力, 但不要把整个 Windows PATH 拼进来 ——
# 那会让 which python 之类的解析随机命中 Windows 的 python.exe。
enabled=true
appendWindowsPath=false
EOF
  ok "已写 /etc/wsl.conf"
  bad "需要 wsl --shutdown 后重进, 再跑一次本脚本"
  [[ $HAVE_SYSTEMD -eq 0 ]] && exit 0
else
  ok "/etc/wsl.conf 已配置"
fi

say "2. 依赖"
NEED=()
for p in python3-venv python3-dev build-essential rsync openssh-client; do
  dpkg -s "$p" &>/dev/null || NEED+=("$p")
done
if (( ${#NEED[@]} )); then
  echo "   安装: ${NEED[*]}"
  sudo apt-get update -qq && sudo apt-get install -y -qq "${NEED[@]}"
fi
ok "依赖齐备   python3 $(python3 --version 2>&1 | cut -d' ' -f2)"

say "3. 目录"
mkdir -p "$SVC"
ok "$SVC"

say "4. 取项目"
if (( SKIP_PULL )); then
  echo "   从本地已同步副本复制 (--skip-pull)"
  rsync -a --info=progress2 \
    --exclude='_server_sync/' --exclude='frontend/node_modules/' \
    --exclude='__pycache__/' --exclude='.git/' \
    "$WIN_PROJ/" "$SVC/"
else
  # 没设就明说, 不要悄悄回退到本地副本 —— 那样拉到的可能是几天前的数据
  [[ -n "$WIN_KEY" && -n "$SRV" ]] || {
    bad "要从服务器拉, 先设 LOL_SERVER (ubuntu@<IP>) 和 LOL_SSH_KEY (/mnt/c/... 密钥路径); 或者加 --skip-pull"
    exit 1; }
  install -d -m 700 "$HOME/.ssh"
  cp "$WIN_KEY" "$HOME/.ssh/amd6x.key"
  chmod 600 "$HOME/.ssh/amd6x.key"          # /mnt/c 上权限位无效, 必须复制进来
  ok "密钥已复制并设 600"
  if ssh -i "$HOME/.ssh/amd6x.key" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
       -o BatchMode=yes "$SRV" true 2>/dev/null; then
    ok "服务器可达, 直接拉最新的"
    rsync -az --info=progress2 -e "ssh -i $HOME/.ssh/amd6x.key -o StrictHostKeyChecking=accept-new" \
      --exclude='__pycache__/' --exclude='node_modules/' \
      "$SRV:/home/ubuntu/lol/service/" "$SVC/"
    scp -q -i "$HOME/.ssh/amd6x.key" "$SRV:/home/ubuntu/lol/service.env" "$ROOT/service.env" || true
  else
    bad "服务器不可达, 回退到本地副本"
    rsync -a --exclude='_server_sync/' --exclude='frontend/node_modules/' \
      --exclude='__pycache__/' --exclude='.git/' "$WIN_PROJ/" "$SVC/"
  fi
fi
[[ -f "$ROOT/service.env" ]] || cp "$SVC/deploy/service.env.example" "$ROOT/service.env"
chmod 600 "$ROOT/service.env"
ok "$(find "$SVC" -type f | wc -l) 个文件, $(du -sh "$SVC" | cut -f1)"
for d in data collected backfill predictions artifacts; do
  n=$(find "$SVC/$d" -type f 2>/dev/null | wc -l)
  printf "      %-12s %4s 个  %s\n" "$d" "$n" "$(du -sh "$SVC/$d" 2>/dev/null | cut -f1)"
done

say "5. venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$SVC/requirements.txt"
ok "$("$VENV/bin/python" --version 2>&1)   $("$VENV/bin/pip" list 2>/dev/null | wc -l) 个包"

say "6. 回归测试 (装单元之前先确认代码能跑)"
if (cd "$SVC" && "$VENV/bin/python" -m unittest discover -s tests 2>&1 | tail -3); then
  ok "测试通过"
else
  bad "测试失败 —— 先别装单元, 查清楚再说"; exit 1
fi

say "7. systemd 单元"
UNITS="$SVC/deploy"
for u in lol-predict.service lol-collect.service lol-collect.timer \
         lol-update.service lol-update.timer lol-backfill.service lol-backfill.timer; do
  [[ -f "$UNITS/$u" ]] || { bad "缺 $u"; continue; }
  sudo sed -e "s#/home/ubuntu/lol#$ROOT#g" \
           -e "s#^User=ubuntu#User=$U#" \
           -e "s#^Group=ubuntu#Group=$U#" \
           -e "s#$ROOT/venv/bin#$VENV/bin#g" \
           -e "s#^MemoryHigh=.*#MemoryHigh=8G#" \
           -e "s#^MemoryMax=.*#MemoryMax=12G#" \
           "$UNITS/$u" | sudo tee "/etc/systemd/system/$u" >/dev/null
done
ok "7 个单元已写入 (路径改写为 $ROOT, 用户 $U, venv 统一到 venv312)"
sudo systemctl daemon-reload

say "8. 启动"
sudo systemctl enable --now lol-predict.service
for t in lol-collect lol-update lol-backfill; do
  sudo systemctl enable --now "$t.timer"
done
ok "已 enable + start"

say "9. 验证"
for i in $(seq 1 60); do
  if curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    ok "/health 就绪 (第 ${i} 次探测)"
    curl -s http://127.0.0.1:8000/health | head -c 400; echo
    break
  fi
  (( i == 60 )) && { bad "60 次探测仍未就绪, 看 journalctl -u lol-predict -n 50"; }
  sleep 5
done
echo
systemctl is-active lol-predict.service | sed 's/^/   lol-predict: /'
systemctl list-timers 'lol-*' --no-pager 2>/dev/null | head -5

cat <<EOF

$(printf '\033[1m== 还差最后一步 (在 Windows 侧做) ==\033[0m')

WSL 不随 Windows 开机启动 —— 不做这步, 重启之后四个单元一个都不会跑,
而且**不会报任何错**。用管理员 PowerShell:

    \$a = New-ScheduledTaskAction -Execute 'wsl.exe' \`
           -Argument '-d Ubuntu-24.04 -u root -e /bin/true'
    \$t = New-ScheduledTaskTrigger -AtLogOn
    \$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries \`
           -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName 'WSL-LoL-Boot' -Action \$a \`
           -Trigger \$t -Settings \$s -RunLevel Highest

访问 UI:  http://127.0.0.1:8000/   (WSL 的 localhost 会转发到 Windows)
EOF
