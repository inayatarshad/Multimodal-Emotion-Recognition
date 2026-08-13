/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark, restrained, scientific. One accent colour, used sparingly.
        ink: {
          950: '#08090c',
          900: '#0d0f14',
          850: '#12151c',
          800: '#181c25',
          700: '#232834',
          600: '#333a4a',
          500: '#4a5265',
        },
        chalk: {
          100: '#f2f4f8',
          200: '#d8dde7',
          300: '#aab2c2',
          400: '#7e8799',
        },
        accent: {
          DEFAULT: '#4dd4c4',
          dim: '#2a8d83',
          glow: 'rgba(77, 212, 196, 0.16)',
        },
        warn: '#e8a33d',
        danger: '#e05c5c',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      transitionTimingFunction: {
        smooth: 'cubic-bezier(0.22, 1, 0.36, 1)',
      },
    },
  },
  plugins: [],
};
