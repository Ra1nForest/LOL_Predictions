#!/usr/bin/env bash
#
# 安装 / 更新 systemd 托管。可以反复跑。
#
#   sudo bash deploy/install.sh
#
# 跑之前先确认三件事:
#   1. 代码在 /home/ubuntu/lol/service/
#   2. venv 在 /home/ubuntu/lol/venv/  且依赖装齐
#   3. artifacts/ 里有五个模型文件 (train.py + ingame_train.py 的产物)
#
set -euo pipefail

ROOT=/home/ubuntu/lol
SVC=$ROOT/service
VENV=$ROOT/venv
ENVF=$ROOT/service.env
UNIT=/etc/systemd/system/lol-predict.service
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
die(){ printf '\n  ✗ %s\n' "$*" >&2; exit 1; }

say "① 检查前提"

[[ -d $SVC ]]                 || die "找不到 $SVC"
[[ -x $VENV/bin/uvicorn ]]    || die "找不到 $VENV/bin/uvicorn —— venv 没建好或依赖没装"
[[ -f $SVC/api.py ]]          || die "找不到 $SVC/api.py"
[[ -d $SVC/static ]]          || die "找不到 $SVC/static —— 前端没上传, 服务会起来但根路径是 404"

missing=()
for m in model_pre_draft model_post_draft model_ingame; do
  [[ -f $SVC/artifacts/$m.json ]] || missing+=("$m.json")
done
if (( ${#missing[@]} )); then
  die "artifacts/ 缺少: ${missing[*]}
     先跑  cd $SVC && $VENV/bin/python train.py && $VENV/bin/python ingame_train.py"
fi
echo "  ✓ 代码、venv、静态目录、模型文件都在"

say "② 环境变量文件"

if [[ ! -f $ENVF ]]; then
  cp "$HERE/service.env.example" "$ENVF"
  chown ubuntu:ubuntu "$ENVF"
  chmod 600 "$ENVF"
  echo "  已从模板创建 $ENVF —— 现在去把密钥填进去, 填完再跑一次本脚本"
  echo "  (没填也能起, 但 /debate 不可用)"
else
  chmod 600 "$ENVF"
  echo "  ✓ $ENVF 已存在, 权限已收到 600"
fi

say "③ 安装 unit"

install -m 644 "$HERE/lol-predict.service" "$UNIT"
systemctl daemon-reload
systemctl enable lol-predict >/dev/null
echo "  ✓ $UNIT 已安装并设为开机自启"

say "④ 重启服务"

# 前台跑着的 uvicorn 会占住 8000, 先清掉, 否则新服务起不来
if pgrep -f "uvicorn api:app" | grep -qv "$(systemctl show -p MainPID --value lol-predict)"; then
  echo "  发现前台运行的 uvicorn, 正在停止"
  pkill -f "uvicorn api:app" || true
  sleep 2
fi

systemctl restart lol-predict

say "⑤ 等待就绪 (冷启动要加载五年 CSV, 通常 30-90 秒)"

for i in $(seq 1 60); do
  if curl -sf -o /dev/null http://127.0.0.1:8000/health; then
    echo "  ✓ 服务已响应 (${i}0 秒内)"
    curl -s http://127.0.0.1:8000/health \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); print("    状态", d["status"], "· 数据截至", d["data_through"])'
    say "完成"
    echo "  日志    journalctl -u lol-predict -f"
    echo "  重启    sudo systemctl restart lol-predict"
    echo "  前端    http://\$(curl -s ifconfig.me):8000/"
    exit 0
  fi
  sleep 10
done

die "60 次探测后仍未就绪。看日志: journalctl -u lol-predict -n 80 --no-pager"
