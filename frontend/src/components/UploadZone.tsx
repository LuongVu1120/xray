"use client";
import { useRef, useState, DragEvent } from "react";
import { Upload, ImageIcon, X } from "lucide-react";
import clsx from "clsx";

interface Props {
  onFile: (f: File) => void;
  disabled?: boolean;
}

export function UploadZone({ onFile, disabled }: Props) {
  const inputRef  = useRef<HTMLInputElement>(null);
  const [drag, setDrag]       = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [name, setName]       = useState<string | null>(null);

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) return;
    setName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
    onFile(file);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDrag(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function clear(e: React.MouseEvent) {
    e.stopPropagation();
    setPreview(null);
    setName(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      className={clsx(
        "relative group cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden",
        "min-h-[260px] flex flex-col items-center justify-center",
        drag ? "border-brand-400 bg-brand-600/10" : "border-slate-700 hover:border-slate-500 bg-slate-800/40",
        disabled && "pointer-events-none opacity-60"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />

      {preview ? (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={preview} alt="preview" className="absolute inset-0 w-full h-full object-contain p-4" />
          <div className="absolute inset-0 bg-slate-950/0 group-hover:bg-slate-950/40 transition-all" />
          {name && (
            <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
              <span className="text-xs text-slate-300 bg-slate-900/80 px-2 py-1 rounded-lg truncate max-w-[80%]">
                {name}
              </span>
              <button onClick={clear} className="bg-slate-900/80 hover:bg-red-500/20 p-1 rounded-lg transition-colors">
                <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
              </button>
            </div>
          )}
          {/* Scanning effect when image present */}
          <div className="scan-line absolute inset-x-0 top-0 h-12 pointer-events-none opacity-50" />
        </>
      ) : (
        <div className="flex flex-col items-center gap-4 text-center p-8">
          <div className={clsx(
            "w-16 h-16 rounded-2xl flex items-center justify-center transition-all",
            drag ? "bg-brand-600/20 scale-110" : "bg-slate-800"
          )}>
            {drag ? (
              <ImageIcon className="w-8 h-8 text-brand-400" />
            ) : (
              <Upload className="w-8 h-8 text-slate-500 group-hover:text-slate-300 transition-colors" />
            )}
          </div>
          <div>
            <p className="font-medium text-slate-300 group-hover:text-white transition-colors">
              Kéo thả hoặc click để chọn ảnh X-quang
            </p>
            <p className="text-sm text-slate-500 mt-1">JPG, PNG, BMP, TIFF — tối đa 10MB</p>
          </div>
          <p className="text-xs text-slate-600 font-mono">
            Hỗ trợ: ảnh X-quang phổi (Chest X-Ray)
          </p>
        </div>
      )}
    </div>
  );
}
