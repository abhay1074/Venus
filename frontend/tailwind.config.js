/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        venus: {
          navy: "#102A43",
          blue: "#2563EB",
          teal: "#0F766E",
          ink: "#1F2937",
          line: "#D8E2EA",
          surface: "#F7FAFC",
        },
      },
      boxShadow: { card: "0 12px 32px rgba(16, 42, 67, 0.08)" },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"] },
    },
  },
  plugins: [],
};
