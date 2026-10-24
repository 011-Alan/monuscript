/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sand: {
          50: '#FDFCFB',
          100: '#FAF8F5',
          200: '#F3EFEA',
          300: '#E7E0D6',
          400: '#D5CABE',
          500: '#BAAB9B',
        },
        monument: {
          terracotta: '#C85A32',
          saffron: '#D97706',
          charcoal: '#1F2923',
          forest: '#263B30',
          ivory: '#FAF8F5',
          gold: '#B45309',
        }
      },
      fontFamily: {
        serif: ['Georgia', 'Cambria', '"Times New Roman"', 'Times', 'serif'],
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
