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

よくある原因:
- Tailscale がログインできていない / 接続先 PC がオフライン
  → 接続先 PC で `tailscale status` を確認。
- `Permission denied (publickey)`
  → 手順A の登録漏れ、または `administrators_authorized_keys` の権限不正。
    手順A を再実行する。
- 鍵が見つからない
  → `config` の `IdentityFile` パスがノートPC上の実パスと一致しているか確認。
