export const metadata = {
  title: "Resume Analyzer",
  description: "ATS Resume Analyzer Frontend",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
