import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Graph Validator Chat',
  description: 'Interactive graph validation chat interface',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

