"""
在桌面生成"双击就能看看板"的启动器 (无窗口)
============================================

为什么用脚本生成而不是直接写文件
--------------------------------
.bat / .vbs 都由 Windows 按**系统 ANSI 编码 (中文 Windows 上是 GBK)** 解析。
用 UTF-8 写的话中文会变成乱码, 而乱码里可能含被解析器当成分隔符的字节 ——
实测一行中文 rem 注释就能把下一条命令吃掉, 报 "'…' 不是内部或外部命令"。

这和项目里 compare.ps1 那个坑同类 (PowerShell 5.1 把 UTF-8 无 BOM 当 ANSI
读, 把变量内容读坏还谎报"完全一致")。凡是给 Windows 原生工具吃的文本,
编码都要显式指定。

为什么是 VBS 不是 BAT
---------------------
ssh -N 不会返回, 用 .bat 的话那个黑窗口必须一直开着 —— 关掉就断线。
WScript.Shell 的 Run 方法第二个参数传 0 就是**完全隐藏窗口**, 而且不像
`start /min` 那样会闪一下。

双击行为:
  · 隧道没开   -> 静默建隧道, 通了自动开浏览器
  · 隧道已开   -> 直接开浏览器 (不重复建)
  · 想断开     -> 双击「断开连接」

用法 (PowerShell):
    $env:LOL_SERVER = "ubuntu@<服务器IP>"
    $env:LOL_SSH_KEY = "C:\\<密钥路径>"
    python tools/make_launcher.py
"""
import os
import sys
from pathlib import Path

DESK = Path.home() / "Desktop"
# 密钥路径和服务器地址不写进代码 (仓库是公开的), 生成时从环境变量取。
# 生成出来的 .vbs 里会带上真实值 —— 它写在桌面上, 不在仓库里
KEY = os.environ.get("LOL_SSH_KEY", "")
SRV = os.environ.get("LOL_SERVER", "")
PORT = 8000

LAUNCH = f"""' LOL 胜率看板 —— 静默启动 SSH 隧道并打开网页
Option Explicit
Dim sh, fso, url, key, tunnelCmd, i, connected
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

url = "http://127.0.0.1:{PORT}/"
key = "{KEY}"

If Not fso.FileExists(key) Then
    MsgBox "找不到密钥文件:" & vbCrLf & key, 16, "LOL 胜率看板"
    WScript.Quit 1
End If

' 已经连着就别再建一条 —— 第二条会因端口冲突失败, 反而把好用的那条搞乱
If PortOpen() Then
    sh.Run url, 1, False
    WScript.Quit 0
End If

' 第二个参数 0 = 完全隐藏窗口; 第三个 False = 不等它结束
' Chr(34) 就是双引号。路径可能含空格, 必须裹起来。
' 这里不直接写 "" 转义, 是因为这段 VBS 由 Python 生成, 连续三个引号会把
' Python 的三引号字符串提前截断 —— 踩过一次。
tunnelCmd = "ssh -i " & Chr(34) & key & Chr(34) & _
            " -o StrictHostKeyChecking=no" & _
            " -o ServerAliveInterval=30 -o ServerAliveCountMax=3" & _
            " -N -L {PORT}:127.0.0.1:{PORT} {SRV}"
sh.Run tunnelCmd, 0, False

' 轮询端口而不是死等固定秒数 —— 网络快慢差很多
connected = False
For i = 1 To 40
    WScript.Sleep 500
    If PortOpen() Then
        connected = True
        Exit For
    End If
Next

If connected Then
    sh.Run url, 1, False
Else
    MsgBox "连不上服务器。" & vbCrLf & vbCrLf & _
           "检查网络, 或在命令行手动试一次看报什么:" & vbCrLf & tunnelCmd, _
           48, "LOL 胜率看板"
End If

' 端口有没有在监听。
'
' 必须用 Run(..., 0, True) 而不是 Exec —— **Exec 一定会弹出一个控制台窗口**,
' 哪怕只闪一下也很难看, 而且这个函数在轮询里要跑几十次。Run 的第二个参数
' 传 0 才是真隐藏, 第三个 True 表示等它结束。代价是拿不到 stdout, 所以
' 转存到临时文件再读。
Function PortOpen()
    Dim q, tmp, cmd, f, out
    q = Chr(34)
    tmp = fso.GetSpecialFolder(2) & "\lolport.txt"
    cmd = "cmd /c netstat -an | findstr " & q & "127.0.0.1:{PORT}" & q & _
          " | findstr LISTENING > " & q & tmp & q
    sh.Run cmd, 0, True
    out = ""
    If fso.FileExists(tmp) Then
        Set f = fso.OpenTextFile(tmp, 1)
        If Not f.AtEndOfStream Then out = f.ReadAll()
        f.Close
    End If
    PortOpen = (InStr(out, "LISTENING") > 0)
End Function
"""

STOP = f"""' 断开 LOL 胜率看板的 SSH 隧道
Option Explicit
Dim wmi, procs, p, n
Set wmi = GetObject("winmgmts:\\\\.\\root\\cimv2")
' 只杀转发这个端口的那条 ssh, 别误伤你自己开的其它 ssh 会话
Set procs = wmi.ExecQuery( _
    "SELECT ProcessId, CommandLine FROM Win32_Process WHERE Name = 'ssh.exe'")
n = 0
For Each p In procs
    If Not IsNull(p.CommandLine) Then
        If InStr(p.CommandLine, "{PORT}:127.0.0.1:{PORT}") > 0 Then
            p.Terminate()
            n = n + 1
        End If
    End If
Next
If n > 0 Then
    MsgBox "已断开 " & n & " 条隧道。", 64, "LOL 胜率看板"
Else
    MsgBox "本来就没连着。", 64, "LOL 胜率看板"
End If
"""


def main():
    if not KEY or not SRV:
        sys.exit("先设环境变量 LOL_SSH_KEY (密钥文件路径) 和 LOL_SERVER (如 ubuntu@<IP>), 见文件头")
    files = [("LOL胜率看板.vbs", LAUNCH), ("断开连接.vbs", STOP)]
    for name, text in files:
        p = DESK / name
        # GBK + CRLF —— 见文件头的说明
        p.write_text(text, encoding="gbk", newline="\r\n")
        print(f"  写入 {p.name}  ({p.stat().st_size} bytes)")

    # 早先的 .bat 版本会挂一个关不掉的黑窗口, 换掉了
    for old in ("LOL胜率看板.bat", "_open_lol.bat"):
        q = DESK / old
        if q.exists():
            q.unlink()
            print(f"  删除旧版 {old}")
    print("\n双击「LOL胜率看板」看板, 双击「断开连接」断开。全程无窗口。")


if __name__ == "__main__":
    main()
