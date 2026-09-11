import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#16202a",
        muted: "#62707d",
        canvas: "#f5f7f8",
        brand: {
          50: "#eaf8f6",
          100: "#d3f1ec",
          500: "#169c8b",
          600: "#087f72",
          700: "#06675d",
        },
      },
      boxShadow: {
        card: "0 16px 45px rgba(22, 32, 42, 0.07)",
      },
    },
  },
  plugins: [],
};

export default config;
