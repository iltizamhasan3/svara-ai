import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SVARA AI",
  description: "Feedback intelligence untuk evaluasi layanan.",
};

export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
