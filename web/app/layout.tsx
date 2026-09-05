import type { Metadata } from "next";
import { Instrument_Sans } from "next/font/google";
import "./globals.css";

// Instrument Sans: a grotesque with slightly narrow, geometric forms and a
// genuinely good tabular figure set — which matters here, because half the
// interface is timestamps and scores that need to align in columns.
const instrumentSans = Instrument_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Gist",
  description:
    "Ask questions about hours of video. Gist scores every frame and audio window against your question and keeps only the few that answer it.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      // Light is the intended look. The .light class wins over the
      // prefers-color-scheme block in globals.css, so a dark OS setting no
      // longer flips the page.
      className={`light ${instrumentSans.variable} h-full antialiased`}
    >
      {/* Extensions (ColorZilla, Grammarly, password managers) stamp attributes
          like cz-shortcut-listen onto <body> before React hydrates, which React
          reports as a mismatch. suppressHydrationWarning applies one level deep
          only, so this silences that specific noise without masking a real
          mismatch inside the tree. */}
      <body
        suppressHydrationWarning
        className="min-h-full flex flex-col bg-background text-foreground"
      >
        {children}
      </body>
    </html>
  );
}
