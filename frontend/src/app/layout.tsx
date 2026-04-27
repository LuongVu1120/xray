import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "AI X-Ray Diagnosis | VuDaiLuong",
  description: "Hệ thống chẩn đoán ảnh X-quang phổi bằng Deep Learning — DenseNet121 + Grad-CAM",
  keywords: ["AI", "X-Ray", "diagnosis", "deep learning", "chest", "pneumonia"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
