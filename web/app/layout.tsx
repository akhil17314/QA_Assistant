import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Document Q&A",
  description: "Ask questions grounded in your uploaded PDFs.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
