# この PC への SSH 接続セットアップ手順

ノートPC(接続元)から、このデスクトップ PC へ **Tailscale 経由・公開鍵認証** で SSH 接続するための手順書。

## 接続先 PC の情報

| 項目 | 値 |
|---|---|
| ホスト名 | `DESKTOP-02D2QIU` |
| ユーザー | `owner`(**管理者アカウント**) |
| Tailscale IP | `100.78.111.87`(MagicDNS 名 `desktop-02d2qiu`) |
| LAN IP | `192.168.0.16`(Wi-Fi) |
| SSH サーバー | OpenSSH Server (sshd) — Running / 自動起動 |
| リッスン | `0.0.0.0:22`, `[::]:22` |

## 重要な前提:管理者ユーザーの鍵ファイル

`owner` は **管理者アカウント**。Windows OpenSSH は管理者ユーザーの場合、
`~/.ssh/authorized_keys` を**無視**し、次のファイルだけを参照する:

```
C:\ProgramData\ssh\administrators_authorized_keys
```

(`sshd_config` の `Match Group administrators` 設定による)
→ 公開鍵は必ずこのファイルに登録すること。

## 使用する鍵

`C:\Users\owner\.ssh` 内の鍵ペアを使用する。

- 秘密鍵: `id_rsa_file_name`
- 公開鍵: `id_rsa_file_name.pub`

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILaafcspdvKJe/J4NMzxZhQZGKfQAUBx2p6dlg5BT+cE comment
```

---

## 手順A:この PC で公開鍵を登録(一度だけ・管理者権限)

`スタートボタンを右クリック → 「ターミナル(管理者)」`(または「Windows PowerShell(管理者)」)を開き、
以下を丸ごと貼り付けて実行する:

```powershell
$key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILaafcspdvKJe/J4NMzxZhQZGKfQAUBx2p6dlg5BT+cE comment'
$f = "$env:ProgramData\ssh\administrators_authorized_keys"
if (-not (Test-Path $f)) { New-Item -ItemType File -Path $f -Force | Out-Null }
if ((Get-Content $f -ErrorAction SilentlyContinue) -notcontains $key) {
    Add-Content -Path $f -Value $key -Encoding ascii
    Write-Host "key added"
} else { Write-Host "key already present" }
icacls $f /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Null
Get-Content $f
```

最後に登録済みの鍵一覧が表示される。`...BT+cE comment` の行があれば登録成功。

> このコマンドは冪等(べきとう)なので、複数回実行しても重複登録されない。

---

## 手順B:ノートPC(接続元)側

1. **Tailscale をインストールし、同じアカウント(hikaru122700)でログイン。**
   これで同じ tailnet に入る(必須)。

   ```powershell
   winget install --id Tailscale.Tailscale -e --accept-source-agreements --accept-package-agreements
   tailscale up   # 表示される URL を hikaru122700 アカウントで承認
   ```

2. `C:\Users\owner\.ssh` をノートPCの自分のユーザーフォルダ
   `C:\Users\<ノートPCのユーザー名>\.ssh` にコピーする。

3. コピーした `.ssh\config` を開き、この PC 用の接続先を追記する。
   `IdentityFile` のパスは **ノートPC上の実際のパス** に直すこと:

   ```
   Host desktop
       HostName 100.78.111.87
       User owner
       IdentityFile C:\Users\<ノートPCのユーザー名>\.ssh\id_rsa_file_name
   ```

4. 接続する:

   ```powershell
   ssh desktop
   ```

   config を使わず直接指定する場合:

   ```powershell
   ssh -i C:\Users\<ノートPCのユーザー名>\.ssh\id_rsa_file_name owner@100.78.111.87
   ```

---

## 手順C:接続先 PC をスリープさせない(外出先接続の必須設定)

**外出先から繋ぐには、留守中もこの PC が起動し続けている必要がある。**
スリープに入ると Tailscale から消え、`tailscale ping` が `rx 0`(無応答)、
SSH は `Connection timed out` になる。

### 症状の見分け方
- `tailscale ping 100.78.111.87` → `timed out` / ピア詳細が `tx NNNN rx 0`
  → 相手 PC がスリープ or 電源オフ。SSH の問題ではない。

### 対処:電源接続時(AC)はスリープ/休止しない

この PC で実行(管理者権限は不要):

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

確認(AC のインデックスが `0x00000000` = なし になっていればOK):

```powershell
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "現在の AC|Current AC"
```

> バッテリ駆動時(DC)は変更しない。据置きで常時 AC 接続のため AC のみ無効化する。
> **2026-06-09 にこの設定を適用済み**(変更前: AC 15分 / DC 10分 でスリープしていた)。

### 運用上の注意
- **外出中は PC の電源を切らない**(物理的に落ちると外出先からは起こせない)。
- 据置きでも `display` は消えてよい(ネットワークには影響しない)。

---

## 注意点

- **秘密鍵 `id_rsa_file_name` はノートPCの外に出さない**(USB・チャット・メールで送らない)。
  コピーは信頼できる手段で行う。
- ノートPCのユーザー名が `owner` 以外なら、`config` の `IdentityFile` のパスを必ず修正する
  (`C:\Users\owner\...` のままだと鍵が見つからない)。
- Tailscale 経由なので、外出先・別ネットワークからでも接続できる(NAT越え・暗号化済み)。
- 同一 LAN 内から `192.168.0.16` で繋ぐ場合は、別途 Windows ファイアウォールに
  22番ポートの受信許可ルールが必要(管理者権限)。Tailscale 経由なら不要。

## トラブルシュート

うまく繋がらない時は、ノートPC側で詳細ログ付きで実行し、出力を確認する:

```powershell
ssh -v desktop
```

| 症状 | 原因 / 対処 |
|---|---|
| `Connection timed out` / `tailscale ping` が `rx 0` | 接続先 PC がスリープ or 電源オフ。**手順C** を適用し、電源を入れたまま外出する。 |
| `Permission denied (publickey)` | 接続先 PC で公開鍵が未登録。**手順A** を実行する。 |
| `Could not resolve hostname` 等 | Tailscale が未ログイン / 相手がオフライン。両端で `tailscale status` を確認。 |
| 鍵が見つからない | `config` の `IdentityFile` パスがノートPC上の実パスと一致しているか確認。 |

### 補足:外出先から家の PC を「起動」したい場合
電源が落ちていると外出先からは起こせない。将来的に外出先から起動したいなら
**Wake-on-LAN(WoL)** の設定が別途必要(BIOS/NIC 設定 + 同一 LAN 内の常時起動機器
or Tailscale 経由 WoL の構成)。
