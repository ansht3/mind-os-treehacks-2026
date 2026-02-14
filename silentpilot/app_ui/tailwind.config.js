/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./styles/**/*.css",
  ],
  theme: {
    extend: {
      colors: {
        'sp-bg': '#0a0a0f',
        'sp-card': '#13131a',
        'sp-border': '#2a2a3a',
        'sp-accent': '#6366f1',
        'sp-green': '#22c55e',
        'sp-red': '#ef4444',
        'sp-yellow': '#eab308',
      },
    },
  },
  plugins: [],
};
