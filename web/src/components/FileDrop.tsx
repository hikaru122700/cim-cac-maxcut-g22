import { useCallback, useRef, useState } from "react";

interface Props {
  label: string;
  hint?: string;
  accept?: string;
  filename?: string | null;
  onFile: (text: string, filename: string) => void;
}

/**
 * ドラッグ & ドロップ + クリック選択に対応した軽量ファイル入力。
 * 読み込みは FileReader でテキストとして一括読み。
 */
export function FileDrop({ label, hint, accept, filename, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      setError(null);
      const reader = new FileReader();
      reader.onload = () => {
        const text = typeof reader.result === "string" ? reader.result : "";
        onFile(text, file.name);
      };
      reader.onerror = () => setError("failed to read file");
      reader.readAsText(file);
    },
    [onFile],
  );

  const onDrop = useCallback(
    (ev: React.DragEvent) => {
      ev.preventDefault();
      setDragActive(false);
      const f = ev.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  return (
    <div
      className={`file-drop${dragActive ? " active" : ""}`}
      onDragOver={(ev) => {
        ev.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(ev) => {
        if (ev.key === "Enter" || ev.key === " ") inputRef.current?.click();
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(ev) => {
          const f = ev.target.files?.[0];
          if (f) handleFile(f);
          // 同じファイルを再選択できるように reset
          ev.target.value = "";
        }}
      />
      <div className="file-drop-label">{label}</div>
      {hint && <div className="file-drop-hint">{hint}</div>}
      {filename && (
        <div className="file-drop-filename">✓ {filename}</div>
      )}
      {error && <div className="file-drop-error">{error}</div>}
    </div>
  );
}
