"use client";
import type { PatientContext } from "@/lib/api";

interface Props {
  value: PatientContext;
  onChange: (next: PatientContext) => void;
  disabled?: boolean;
}

export function PatientForm({ value, onChange, disabled }: Props) {
  return (
    <div className="grid sm:grid-cols-3 gap-3">
      <label className="text-xs text-slate-400 space-y-1">
        <span className="block label">Tuổi</span>
        <input
          type="number"
          min={0}
          max={130}
          disabled={disabled}
          value={value.age ?? ""}
          onChange={(e) =>
            onChange({
              ...value,
              age: e.target.value ? Number(e.target.value) : undefined,
            })
          }
          className="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-brand-500 focus:outline-none disabled:opacity-60"
          placeholder="65"
        />
      </label>
      <label className="text-xs text-slate-400 space-y-1">
        <span className="block label">Giới tính</span>
        <select
          disabled={disabled}
          value={value.sex ?? ""}
          onChange={(e) =>
            onChange({
              ...value,
              sex: (e.target.value as PatientContext["sex"]) || undefined,
            })
          }
          className="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-brand-500 focus:outline-none disabled:opacity-60"
        >
          <option value="">— Không rõ —</option>
          <option value="male">Nam</option>
          <option value="female">Nữ</option>
          <option value="other">Khác</option>
        </select>
      </label>
      <label className="text-xs text-slate-400 space-y-1 sm:col-span-1">
        <span className="block label">Triệu chứng (tuỳ chọn)</span>
        <input
          type="text"
          disabled={disabled}
          value={value.symptoms ?? ""}
          onChange={(e) => onChange({ ...value, symptoms: e.target.value || undefined })}
          className="w-full bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-brand-500 focus:outline-none disabled:opacity-60"
          placeholder="ho 3 tuần, sốt nhẹ"
        />
      </label>
    </div>
  );
}
